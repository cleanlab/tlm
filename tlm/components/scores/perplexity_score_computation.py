import numpy as np

from tlm.components import Component
from tlm.types import Completion


class PerplexityScoreComputation(Component):
    async def execute(self) -> None:
        reference_completions: list[Completion] = self.execution_context.get("reference_completions")

        perplexity_scores = np.array([completion.perplexity for completion in reference_completions])

        self.execution_context.add("perplexity_scores", perplexity_scores)
        self.execution_context.add("use_perplexity_score", any(score is not None for score in perplexity_scores))
