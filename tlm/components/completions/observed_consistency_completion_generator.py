import asyncio

from tlm.components import Component
from tlm.config.presets import ReasoningEffort
from tlm.utils.completion_utils import generate_completion
from tlm.utils.response_format_utils import add_explanation_to_response_format
from tlm.templates import ObservedConsistencyQACompletionTemplate
from tlm.utils.prompt_utils import extract_user_prompt
from tlm.types import Completion, ExtractedResponseField, CompletionParams
from tlm.config.defaults import get_settings
from tlm.config.presets import REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS


settings = get_settings()


class ObservedConsistencyCompletionGenerator(Component):
    def __init__(
        self,
        completion_params: CompletionParams,
        count: int,
        temperature: float,
        reasoning_effort: ReasoningEffort,
        constrain_outputs: list[str] | None,
        depends_on: list[Component] | None = None,
    ):
        if count < 0:
            raise ValueError("count must be non-negative")

        modified_params = add_explanation_to_response_format(completion_params)
        if modified_params:
            completion_params = modified_params

        self.completion_params = completion_params
        self.count = count
        self.temperature = temperature
        self.constrain_outputs = constrain_outputs
        self.reasoning_effort = reasoning_effort
        self.max_explanation_words = REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS[reasoning_effort]
        self.template = ObservedConsistencyQACompletionTemplate.create(
            reasoning_effort=reasoning_effort,
            constrain_outputs=constrain_outputs,
            extract_answer=modified_params is not None,
        )

        super().__init__(depends_on=depends_on)

    async def execute(self) -> None:
        observed_consistency_answers: list[str | None] = []
        observed_consistency_completions = []

        if self.count > 0:
            user_prompt = extract_user_prompt(self.completion_params)
            observed_consistency_completions = await asyncio.gather(
                *[
                    asyncio.create_task(
                        generate_completion(
                            self.template,
                            completion_params=self.completion_params,
                            template_kwargs={
                                "question": user_prompt,
                                "max_explanation_words": self.max_explanation_words,
                            },
                        )
                    )
                    for _ in range(self.count)
                ]
            )

            for completion in observed_consistency_completions:
                if isinstance(completion, Completion):
                    answer = completion.response_fields.get(
                        ExtractedResponseField.ANSWER
                    ) or completion.response_fields.get(ExtractedResponseField.MESSAGE, completion.message)

                    observed_consistency_answers.append(answer)
                else:
                    observed_consistency_answers.append(None)

        self.execution_context.add("consistency_answers", observed_consistency_answers)
        self.execution_context.add("consistency_completions", observed_consistency_completions)
