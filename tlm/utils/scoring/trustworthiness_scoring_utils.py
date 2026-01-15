from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import logging

from tlm.config.presets import WorkflowType
from tlm.config.score_weights import (
    COMPONENT_SCORE_WEIGHTS,
    DEFAULT_MODEL,
    PERPLEXITY_SCORE_WEIGHT,
)

logger = logging.getLogger(__name__)


class WeightedScore:
    """Score part with weight. Weight used when calculating total score out of score parts."""

    def __init__(self, score: np.float64 | None, weight: float):
        self.score = score
        self.weight = weight


def get_trustworthiness_scores(
    workflow_type: WorkflowType,
    model: str,
    consistency_scores: npt.NDArray[np.float64],
    indicator_scores: npt.NDArray[np.float64],
    self_reflection_scores: npt.NDArray[np.float64],
    perplexity_scores: npt.NDArray[np.float64],
    use_perplexity_score: bool,
    prompt_eval_scores: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    return _generate_total_scores(
        consistency_scores=consistency_scores,
        indicator_scores=indicator_scores,
        self_reflection_scores=self_reflection_scores,
        perplexity_scores=perplexity_scores,
        prompt_eval_scores=prompt_eval_scores,
        use_perplexity_score=use_perplexity_score,
        workflow_type=workflow_type,
        model=model,
    )


def _generate_total_scores(
    consistency_scores: npt.NDArray[np.float64],
    indicator_scores: npt.NDArray[np.float64],
    self_reflection_scores: npt.NDArray[np.float64],
    perplexity_scores: npt.NDArray[np.float64],
    prompt_eval_scores: npt.NDArray[np.float64] | None,
    use_perplexity_score: bool,
    workflow_type: WorkflowType,
    model: str,
) -> npt.NDArray[np.float64]:
    """Generates total score for each reference answer (row) in scores dataframe.

    The weights used to calculate total score are different depending on if prompt or get_trustworthiness_score is called and perplexity score is calculated or not.

    If just self reflection score couldn't be computed (value is nan), that value is omitted from the total score calculation.
    If just observed consistency score couldn't be computed (value is nan), that value is omitted from the total score calculation.
    If both self reflection score and observed consistency score are nan, then we want to omit score from totals.

    Note: this is an unoptimized for-loop implementation. Runtime shouldn't be a concern due to small size, and vectorized solution was much more complex.
    """
    # Fill NaN values if consistency completions were disabled
    if len(consistency_scores) == 0:
        consistency_scores = np.full(len(self_reflection_scores), np.nan)
    if len(indicator_scores) == 0:
        indicator_scores = np.full(len(self_reflection_scores), np.nan)

    # Handle case where prompt_eval_scores is None
    if not prompt_eval_scores:
        prompt_eval_scores = np.full(len(self_reflection_scores), np.nan)

    total_scores: List[float] = []

    logger.info("Generating trustworthiness scores with scores:")
    logger.info(f"-- Consistency scores: {consistency_scores}")
    logger.info(f"-- Indicator scores: {indicator_scores}")
    logger.info(f"-- Self reflection scores: {self_reflection_scores}")
    logger.info(f"-- Perplexity scores: {perplexity_scores}")
    logger.info(f"-- Prompt eval scores: {prompt_eval_scores}")

    scores = pd.DataFrame(
        {
            "consistency_score": consistency_scores,
            "indicator_score": indicator_scores,
            "self_reflection_score": self_reflection_scores,
            "perplexity_score": perplexity_scores,
            "prompt_eval_score": prompt_eval_scores,
        }
    )

    (
        consistency_score_weight,
        indicator_score_weight,
        self_reflection_score_weight,
        prompt_eval_score_weight,
        perplexity_score_weight,
    ) = get_score_weights(
        use_perplexity_score=use_perplexity_score,
        workflow_type=workflow_type,
        model=model,
    ).values()

    for _, row in scores.iterrows():
        weighted_score_parts: List[WeightedScore] = [
            WeightedScore(
                score=row["consistency_score"],
                weight=consistency_score_weight,
            ),
            WeightedScore(
                score=row["indicator_score"],
                weight=indicator_score_weight,
            ),
            WeightedScore(
                score=row["self_reflection_score"],
                weight=self_reflection_score_weight,
            ),
            WeightedScore(
                score=row["prompt_eval_score"],
                weight=prompt_eval_score_weight,
            ),
            WeightedScore(
                score=row["perplexity_score"],
                weight=perplexity_score_weight,
            ),
        ]

        active_weighted_score_parts = [
            (part.score, part.weight)
            for part in weighted_score_parts
            if part.score is not None and not np.isnan(part.score)
        ]

        # return NaN if no subscores exist
        if len(active_weighted_score_parts) == 0:
            total_scores.append(np.nan)
        else:
            total_scores.append(
                float(
                    np.average(
                        [part[0] for part in active_weighted_score_parts],
                        weights=[part[1] for part in active_weighted_score_parts],
                    )
                )
            )

    return np.array(total_scores)


def get_score_weights(use_perplexity_score: bool, workflow_type: WorkflowType, model: str) -> Dict[str, float]:
    """Determines which weights to use for the total score calculation and returns appropriate weights dictionary.
    Weights are dependent on the model used and if we have a perplexity score calculated.
    """

    # First get weights for the current workflow type
    workflow_weights = COMPONENT_SCORE_WEIGHTS.get(workflow_type, COMPONENT_SCORE_WEIGHTS[WorkflowType.DEFAULT])

    # Then get model-specific weights for this workflow type
    score_weights = workflow_weights.get(model, workflow_weights[DEFAULT_MODEL]).copy()

    if not use_perplexity_score:
        score_weights[PERPLEXITY_SCORE_WEIGHT] = np.nan

    return score_weights
