import numpy as np
import logging

from tlm.components import Component
from tlm.config.presets import WorkflowType
from tlm.utils.scoring.trustworthiness_scoring_utils import get_trustworthiness_scores

logger = logging.getLogger(__name__)


class TrustworthinessScoreComputation(Component):
    def __init__(
        self,
        workflow_type: WorkflowType,
        model: str,
        depends_on: list[Component] | None = None,
    ):
        self.workflow_type = workflow_type
        self.model = model
        super().__init__(depends_on=depends_on)

    async def execute(self):
        consistency_scores = np.array(self.execution_context.get("consistency_scores"), dtype=np.float64)
        indicator_scores = np.array(self.execution_context.get("indicator_scores"), dtype=np.float64)
        self_reflection_scores = np.array(self.execution_context.get("self_reflection_scores"), dtype=np.float64)
        perplexity_scores = self.execution_context.get("perplexity_scores")
        use_perplexity_score = self.execution_context.get("use_perplexity_score")
        prompt_evaluation_scores = self.execution_context.get("prompt_evaluation_scores", [])

        trustworthiness_scores = get_trustworthiness_scores(
            self.workflow_type,
            self.model,
            consistency_scores,
            indicator_scores,
            self_reflection_scores,
            perplexity_scores,
            use_perplexity_score,
            prompt_evaluation_scores,
        )

        logger.info(f"Calculated trustworthiness scores: {trustworthiness_scores}")

        self.execution_context.add("trustworthiness_scores", trustworthiness_scores)
