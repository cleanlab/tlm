import numpy as np
import numpy.typing as npt
import pytest

from tlm.utils.math_utils import (
    ASYMPTOTIC_EPSILON,
    compute_cosine_similarity,
    get_median_indices,
    get_nan_safe_mean,
    harmonic_mean,
    make_score_asymptotic,
)


@pytest.mark.parametrize(
    "a, b, expected, description",
    [
        ([1, 0], [1, 0], 1, "Two identical vectors"),
        ([1, 0], [0, 1], 0, "Two orthogonal vectors"),
        ([1, 0], [2, 0], 1, "Two vectors with different magnitudes"),
        ([1, 0], [-1, 0], 0, "Two opposite vectors (clipping at 0)"),
        ([1, 0], [3, 4], 0.6, "Two vectors with different magnitudes and different directions"),
        ([1, 0, 0], [0, 1, 0], 0, "Two orthogonal vectors"),
        ([1, 0, 0], [0, 0, 1], 0, "Two orthogonal vectors"),
        ([1, 0, 0], [1, 0, 0], 1, "Two identical vectors"),
        ([1, 0, 0], [-1, 0, 0], 0, "Two opposite vectors (clipping at 0)"),
        ([1, 0, 0], [2, 0, 0], 1, "Two congruent vectors with different magnitudes"),
        ([1, 0, 0], [3, 0, 4], 0.6, "Two vectors with different magnitudes and different directions"),
    ],
)
def test_compute_cosine_similarity(a: list[float], b: list[float], expected: float, description: str) -> None:
    assert compute_cosine_similarity(a, b) == expected


@pytest.mark.parametrize(
    "scores_matrix, expected",
    [
        (np.array([[2, 3, 1]]), np.array([0])),
        (np.array([[2, 3, 1], [4, 5, 6]]), np.array([0, 1])),
        (np.array([[2, 3, 1, np.nan], [np.nan, np.nan, 0, np.nan], [np.nan, 4, 5, 6]]), np.array([0, 2, 2])),
    ],
)
def test_get_median_indices(scores_matrix: npt.NDArray[np.float64], expected: npt.NDArray[np.int_]) -> None:
    assert np.all(get_median_indices(scores_matrix) == expected)


@pytest.mark.parametrize(
    "scores, axis, expected_array_length, expected",
    [
        (np.array([[2, 3, 1]]), 0, None, np.array([2, 3, 1])),
        (np.array([[2, 3, 1]]), 1, None, np.array([2])),
        (np.array([[2, 3, 1], [4, 5, 6]]), 0, None, np.array([3, 4, 3.5])),
        (np.array([[2, 3, 1], [4, 5, 6]]), 1, None, np.array([2, 5])),
        (
            np.array([[2, 3, 1, np.nan], [np.nan, np.nan, 0, np.nan], [np.nan, 4, 5, 6]]),
            0,
            None,
            np.array([2, 3.5, 2, 6]),
        ),
        (np.array([[2, 3, 1, np.nan], [np.nan, np.nan, 0, np.nan], [np.nan, 4, 5, 6]]), 1, None, np.array([2, 0, 5])),
        (np.array([[np.nan, np.nan, np.nan]]), 0, None, np.array([np.nan])),
        (np.array([[np.nan, np.nan, np.nan]]), 1, None, np.array([np.nan])),
        (np.array([[np.nan, np.nan, np.nan]]), 0, 1, np.array([np.nan])),
        (np.array([[np.nan, np.nan, np.nan]]), 1, 1, np.array([np.nan])),
        (np.array([[np.nan, np.nan, np.nan]]), 0, 2, np.array([np.nan, np.nan])),
        (np.array([[np.nan, np.nan, np.nan]]), 1, 2, np.array([np.nan, np.nan])),
    ],
)
def test_get_nan_safe_mean(
    scores: npt.NDArray[np.float64], axis: int, expected_array_length: int, expected: npt.NDArray[np.float64]
) -> None:
    res = get_nan_safe_mean(scores, axis, expected_array_length)
    assert np.all((res == expected) | (np.isnan(res) & np.isnan(expected)))


def test_make_score_asymptotic() -> None:
    for score in range(1000):
        assert ASYMPTOTIC_EPSILON <= make_score_asymptotic(score / 1000) <= 1 - ASYMPTOTIC_EPSILON


def test_harmonic_mean_empty_list() -> None:
    assert harmonic_mean([]) == 0
