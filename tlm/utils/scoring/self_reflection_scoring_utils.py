import logging
from typing import Sequence

import numpy as np
import numpy.typing as npt

from tlm.config.defaults import get_settings
from tlm.utils.math_utils import get_nan_safe_mean
from tlm.types import (
    Completion,
    CompletionFailure,
    CompletionFailureType,
    ExtractedResponseField,
)
from tlm.utils.parse_utils import get_choice_token_confidence

settings = get_settings()
logger = logging.getLogger(__name__)


def generate_self_reflection_scores(
    reference_answers: Sequence[str],
    self_reflection_completions: Sequence[Completion | CompletionFailure],
) -> npt.NDArray[np.float64]:
    """Returns self reflection score for each reference answer."""

    self_reflection_scores_raw = np.array(
        [
            _generate_self_reflection_score(reflection_completion)
            for reflection_completion in self_reflection_completions
        ]
    )

    self_reflection_scores = self_reflection_scores_raw.reshape(
        len(reference_answers), -1, order="F"
    )  # Second dimension size is number of unique self_reflection templates

    # get self reflection score for each reference answer
    return get_nan_safe_mean(
        self_reflection_scores,
        axis=1,
        expected_array_length=len(reference_answers),
    )


def _generate_self_reflection_score(
    reflection_completion: Completion | CompletionFailure,
) -> float:
    """Gets self reflection score given answer.

    If completion failed, reflection answer will be one of API_ERROR, SOFT_TIMEOUT, HARD_TIMEOUT.
    For those cases, we want to omit the answer from the score (so setting score to `np.nan`)

    If the completion succeeded, we will use the answer token confidence score (logprobs) if it's available
    and the config uses logprobs. For logprobs, if the answer is in positive_answers, we use the token confidence
    directly; otherwise we use 1 - token confidence.

    We use a parser to get the answer and using a mapping to get the score.
    If the parser fails, ie. the LLM did not give an expected response
    (neither A nor B in the correctness prompt, or not an integer in the confidence prompt),
    we will return NaN.
    """
    if isinstance(reflection_completion, CompletionFailure):
        if reflection_completion.type == CompletionFailureType.PARSE:
            return settings.SELF_REFLECTION_PARSE_FAILURE_SCORE
        else:
            return np.nan

    scored_answer_token_confidence = get_choice_token_confidence(reflection_completion)
    mapped_score = reflection_completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE)

    # NOTE: skipping abstained answer check because it appears to be unused in current SaaS

    # If have logprobs and config uses them, use the token confidence with positive_answers logic
    if scored_answer_token_confidence is not None:
        return scored_answer_token_confidence
    elif mapped_score is not None:
        return mapped_score
    else:
        return np.nan
