import numpy as np
import numpy.typing as npt

from tlm.utils.math_utils import get_nan_safe_mean


def compute_indicator_scores(
    reference_answers: list[str], comparison_answers: list[str]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns ratio of comparison answers that are identical to reference answer, for each ref answer."""

    indicator_mask = (
        np.array(reference_answers).reshape((len(reference_answers), 1))
        == np.array(comparison_answers).reshape((1, len(comparison_answers)))
    ).astype(np.float64)

    return get_nan_safe_mean(
        indicator_mask,
        axis=1,
        expected_array_length=len(reference_answers),
    ), indicator_mask.flatten()
