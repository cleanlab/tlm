from tlm.components import Component
from tlm.utils.scoring.prompt_evaluation_scoring_utils import get_prompt_evaluation_scores


class PromptEvaluationScoreExtraction(Component):
    async def execute(self) -> None:
        reference_answers = self.execution_context.get("reference_answers")
        prompt_evaluation_completions = self.execution_context.get("prompt_evaluation_completions")

        prompt_evaluation_scores = get_prompt_evaluation_scores(reference_answers, prompt_evaluation_completions)

        self.execution_context.add("prompt_evaluation_scores", prompt_evaluation_scores)
