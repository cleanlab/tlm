import asyncio

from tlm.components import Component
from tlm.templates import PromptAnswerabilityCompletionTemplate
from tlm.utils.completion_utils import generate_completion


class PromptEvaluationCompletionGenerator(Component):
    def __init__(self, prompt: str, temperature: float | None, **kwargs):
        self.prompt = prompt
        self.temperature = temperature
        self.template = PromptAnswerabilityCompletionTemplate.create()
        super().__init__(**kwargs)

    async def execute(self) -> None:
        prompt_evaluation_completions = []

        prompt_evaluation_completion_tasks = [
            asyncio.create_task(
                generate_completion(
                    template=self.template,
                    template_kwargs={
                        "prompt": self.prompt,
                    },
                    temperature=self.temperature,
                )
            )
        ]

        prompt_evaluation_completions = await asyncio.gather(*prompt_evaluation_completion_tasks)
        self.execution_context.add("prompt_evaluation_completions", prompt_evaluation_completions)
