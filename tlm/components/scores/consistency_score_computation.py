from tlm.components import Component
from tlm.utils.scoring.consistency_scoring_utils import (
    compute_consistency_scores,
    compute_consistency_scores_classification,
)
from tlm.utils.scoring.indicator_scoring_utils import compute_indicator_scores
from tlm.types import SimilarityMeasure

import numpy as np


class ConsistencyScoreComputation(Component):
    def __init__(
        self,
        similarity_measure: SimilarityMeasure,
        structured_outputs: bool,
        constrain_outputs: list[str] | None,
        depends_on: list[Component] | None = None,
    ):
        self.similarity_measure = similarity_measure
        self.structured_outputs = structured_outputs
        self.constrain_outputs = constrain_outputs

        super().__init__(depends_on=depends_on)

    async def execute(self):
        reference_answers = self.execution_context.get("reference_answers")
        consistency_answers = self.execution_context.get("consistency_answers")
        consistency_completions = self.execution_context.get("consistency_completions")

        if len(consistency_answers) > 0:
            if self.constrain_outputs is not None:
                average_consistency_scores, consistency_scores_flat = compute_consistency_scores_classification(
                    reference_answers, consistency_answers, consistency_completions, self.constrain_outputs
                )
                # Classification tasks don't use indicator scores
                average_indicator_scores = np.array([None] * len(reference_answers))
                indicator_scores_flat = np.array([None] * len(reference_answers) * len(consistency_answers))
            else:
                average_consistency_scores, consistency_scores_flat = await compute_consistency_scores(
                    reference_answers, consistency_answers, self.similarity_measure, self.structured_outputs
                )
                average_indicator_scores, indicator_scores_flat = compute_indicator_scores(
                    reference_answers, consistency_answers
                )
        else:
            average_consistency_scores = np.array([])
            consistency_scores_flat = np.array([])
            average_indicator_scores = np.array([])
            indicator_scores_flat = np.array([])

        self.execution_context.add("consistency_scores", average_consistency_scores)
        self.execution_context.add("indicator_scores", average_indicator_scores)
        self.execution_context.add("consistency_scores_flat", consistency_scores_flat)
        self.execution_context.add("indicator_scores_flat", indicator_scores_flat)
