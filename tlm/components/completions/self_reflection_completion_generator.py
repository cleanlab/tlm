import asyncio

from tlm.components import Component
from tlm.config.presets import REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS, ReasoningEffort, WorkflowType
from tlm.templates.reflection_completion_templates import SELF_REFLECTION_TEMPLATES_BY_WORKFLOW
from tlm.utils.completion_utils import generate_completion


class SelfReflectionCompletionGenerator(Component):
    def __init__(
        self,
        prompt: str,
        reasoning_effort: ReasoningEffort,
        workflow_type: WorkflowType,
        num_completions: int,
        **kwargs,
    ):
        self.prompt = prompt
        self.reasoning_effort = reasoning_effort

        completion_templates = SELF_REFLECTION_TEMPLATES_BY_WORKFLOW.get(workflow_type, [])
        if len(completion_templates) == 0 and workflow_type == WorkflowType.BINARY_CLASSIFICATION:
            completion_templates = SELF_REFLECTION_TEMPLATES_BY_WORKFLOW[WorkflowType.CLASSIFICATION]

        if num_completions > 1:
            completion_templates = completion_templates[:num_completions]

        self.completion_templates = completion_templates

        super().__init__(**kwargs)

    async def execute(self) -> None:
        reference_answers: list[str] = self.execution_context.get("reference_answers")

        completion_tasks = []

        for answer in reference_answers:
            template_kwargs = {
                "question": self.prompt,
                "answer": answer,
                "max_explanation_words": REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS[self.reasoning_effort],
            }

            completion_tasks.extend(
                [
                    asyncio.create_task(
                        generate_completion(
                            template=template.create(reasoning_effort=self.reasoning_effort),
                            template_kwargs=template_kwargs,
                            temperature=0.0,
                            response_format_model=template.construct_response_format(answer),
                            reference_answer=answer,
                        )
                    )
                    for template in self.completion_templates
                ]
            )

        self_reflection_completions_flat = await asyncio.gather(*completion_tasks)
        # Reshape into 2D array: rows = number of reference answers, cols = number of completion templates
        num_templates = len(self.completion_templates)
        self_reflection_completions = [
            self_reflection_completions_flat[i : i + num_templates]
            for i in range(0, len(self_reflection_completions_flat), num_templates)
        ]
        self.execution_context.add("self_reflection_completions", self_reflection_completions)
