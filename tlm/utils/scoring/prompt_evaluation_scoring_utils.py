from collections.abc import Sequence
from typing import List
import numpy as np
import numpy.typing as npt

from tlm.types import Completion, CompletionFailure, ExtractedResponseField
from tlm.utils.parse_utils import get_choice_token_confidence


def get_prompt_evaluation_scores(
    reference_answers: Sequence[str],
    completions: List[Completion | CompletionFailure],
) -> npt.NDArray[np.float64]:
    scores = _extract_prompt_evaluation_scores(completions)

    # Return array of mean scores with length matching reference_answers
    mean_score = np.mean([s for s in scores if not np.isnan(s)])
    return np.full(len(reference_answers), mean_score, dtype=np.float64)


def _extract_prompt_evaluation_scores(
    completions: List[Completion | CompletionFailure],
) -> list[float]:
    """Returns prompt evaluation scores for each completion.

    Args:
        completions: List of completion dictionaries containing parsed answers

    Returns:
        List of scores for each completion
    """

    # If no completions, return empty list
    if not completions:
        return []

    return [_get_prompt_evaluation_score(completion) for completion in completions]


def _get_prompt_evaluation_score(
    completion: Completion | CompletionFailure,
) -> float:
    """Gets prompt evaluation score for a single completion.

    If completion failed, returns 0.0.
    If the completion succeeded and the parsed answer is in config's positive_answers list, uses the answer token confidence score (logprobs) if available,
    otherwise uses the parsed score from the config's score mapping.

    Args:
        completion: The completion to score

    Returns:
        float score between 0 and 1
    """
    if isinstance(completion, CompletionFailure):
        return 0.0

    # Use token confidence if available
    if scored_answer_token_confidence := get_choice_token_confidence(completion):
        return scored_answer_token_confidence

    # Fall back to mapped score
    if mapped_score := completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE):
        return mapped_score

    return np.nan
