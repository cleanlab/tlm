from typing import List, overload

import numpy as np
import numpy.typing as npt

ASYMPTOTIC_EPSILON = 1e-3  # Small value to provide asymptotic behavior at both ends
ASYMPTOTIC_SCALE = 1 - 2 * ASYMPTOTIC_EPSILON  # Scale factor to ensure range [0, 1] maps to [_EPSILON, 1 - _EPSILON]


def compute_cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.
    Returns a value between 0 and 1 (clipping at 1 to prevent floating point errors,
    clipping at 0 as high dimensional text embeddings are unlikely to produce negative cosine similarity, but just in case).
    """
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(cosine_similarity, 0, 1))


def get_median_indices(scores_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    """Returns indices of median values for each row in the 2D scores array.

    Args:
        scores_matrix: 2D array where each row contains scores (may include NaN values)

    Returns:
        1D array containing the index of the median non-NaN value for each row
    """
    median_indices = np.full(len(scores_matrix), -1)
    for i, scores in enumerate(scores_matrix):
        valid_indices = np.where(~np.isnan(scores))[0]
        valid_scores: npt.NDArray[np.float64] = scores[valid_indices]  # type: ignore[index]
        median_indices[i] = valid_indices[np.argsort(valid_scores)[len(valid_scores) // 2]]
    return median_indices


def get_nan_safe_mean(
    scores: npt.NDArray[np.float64], axis: int = 1, expected_array_length: int | None = None
) -> npt.NDArray[np.float64]:
    """Returns nan if all values in array are nans (or an array of nans if expected_array_length is specified),
    otherwise returns mean along axis of all non-nan values in scores since
    np.nanmean() raises a warning if all values in array are nans."""
    if not np.isnan(scores).all():
        return np.asarray(np.nanmean(scores, axis=axis))

    if expected_array_length is not None:
        return np.full(expected_array_length, np.nan)
    else:
        return np.array([np.nan])


@overload
def _logprob_to_probability(logprob: float) -> float: ...


@overload
def _logprob_to_probability(logprob: List[float]) -> npt.NDArray[np.float64]: ...


def _logprob_to_probability(logprob: List[float] | float) -> npt.NDArray[np.float64] | float:
    """Converts logprob to probability 0-1 scale."""
    return np.exp(logprob)


def make_score_asymptotic(
    score: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Transform score to asymptotic range [epsilon, 1 - epsilon].

    This transformation assumes that the original score is in the range [0, 1].
    """

    return ASYMPTOTIC_EPSILON + ASYMPTOTIC_SCALE * score


def harmonic_mean(scores: List[float]) -> float:
    """Computes the harmonic mean of a list of scores.
    Epsilon is added to each score to avoid zero-division.
    """
    if not scores:
        return 0.0

    epsilon = 1e-3
    return min(1, len(scores) / sum(1 / (s + epsilon) for s in scores))
