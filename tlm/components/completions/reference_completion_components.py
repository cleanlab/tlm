import asyncio
from typing import Any, Dict

from tlm.components import Component
from tlm.config.presets import REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS, ReasoningEffort
from tlm.templates.reference_completion_template import ReferenceCompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField, CompletionParams
from tlm.utils.prompt_utils import extract_user_prompt
from tlm.utils.response_format_utils import add_explanation_to_response_format


class ReferenceCompletionFormatter(Component):
    """
    Used in scoring workflows when reference completions are provided as input.
    This component adds required context for usage by future components.
    """

    def __init__(
        self,
        completion_params: CompletionParams,
        response_input: Dict[str, Any],
        depends_on: list[Component] | None = None,
    ):
        self.completion_params = completion_params

        reference_completion = Completion.from_completion_dict(response_input)
        self.reference_completions = [reference_completion]
        self.reference_answers = [reference_completion.response_fields[ExtractedResponseField.ANSWER]]

        super().__init__(depends_on=depends_on)

    async def execute(self) -> None:
        # this is used for counting input tokens, revisit later
        self.execution_context.add("prompt", extract_user_prompt(self.completion_params))
        self.execution_context.add("reference_completions", self.reference_completions)
        self.execution_context.add("reference_answers", self.reference_answers)


class ReferenceCompletionGenerator(Component):
    def __init__(
        self,
        count: int,
        min_count: int,
        completion_params: CompletionParams,
        alternate_temperature: float | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        constrain_outputs: list[str] | None = None,
        depends_on: list[Component] | None = None,
    ):
        if count <= 0:
            raise ValueError("count must be positive")

        self.count = count
        self.min_count = min(count, min_count)
        self.alternate_temperature = alternate_temperature

        modified_params = add_explanation_to_response_format(completion_params)
        if modified_params:
            completion_params = modified_params
        self.completion_params = completion_params

        self.constrain_outputs = constrain_outputs
        self.max_explanation_words = REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS[reasoning_effort]
        self.template = ReferenceCompletionTemplate.create(
            reasoning_effort=reasoning_effort,
            constrain_outputs=constrain_outputs,
            extract_answer=modified_params is not None,
        )

        super().__init__(depends_on=depends_on)

    async def execute(self) -> None:
        # this is used for counting input tokens, revisit later
        self.execution_context.add("prompt", extract_user_prompt(self.completion_params))

        template_kwargs = {
            "prompt": extract_user_prompt(self.completion_params),
            "max_explanation_words": self.max_explanation_words,
        }

        completion_tasks = [
            asyncio.create_task(
                generate_completion(
                    template=self.template,
                    completion_params=self.completion_params,
                    template_kwargs=template_kwargs,
                    temperature=0.0 if i == 0 else (self.alternate_temperature or 0.0),
                )
            )
            for i in range(self.count)
        ]
        reference_completions = await asyncio.gather(*completion_tasks)

        reference_answers = []
        reference_failures = []

        for result in reference_completions:
            if isinstance(result, Completion):
                answer = result.response_fields.get(ExtractedResponseField.ANSWER) or result.response_fields.get(
                    ExtractedResponseField.MESSAGE, result.message
                )
                reference_answers.append(answer)
            else:
                reference_failures.append(result)

        if len(reference_answers) < self.min_count:
            raise Exception("Not enough reference completions")

        reference_completions = [c for c in reference_completions if isinstance(c, Completion)]

        self.execution_context.add("reference_answers", reference_answers)
        self.execution_context.add("reference_completions", reference_completions)
        self.execution_context.add("reference_failures", reference_failures)
