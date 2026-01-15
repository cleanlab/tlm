import numpy as np
import numpy.typing as npt

from tlm.types import Completion, ExtractedResponseField
from tlm.config.defaults import get_settings

defaults = get_settings()

OBSERVED_CONSISTENCY_EXPLANATION_TEMPLATE = "This response is untrustworthy due to lack of consistency in possible responses from the model. Here's one inconsistent alternate response that the model considered (which may not be accurate either): \n{observed_consistency_completion}"


def get_explainability_message(
    average_trustworthiness_score: float | None,
    self_reflection_completions: list[list[Completion]],
    observed_consistency_completions: list[Completion],
    average_consistency_score: float,
    consistency_scores_flat: npt.NDArray[np.float64],
    best_answer_idx: int,
    best_answer: str,
) -> str:
    explainability_message = ""

    if average_trustworthiness_score is None:
        return explainability_message

    if (
        not np.isnan(average_trustworthiness_score)
        and average_trustworthiness_score < defaults.EXPLAINABILITY_THRESHOLD
    ):
        self_reflection_completions_flat = [
            completion for sublist in self_reflection_completions for completion in sublist
        ]
        average_self_reflection_score = np.mean(
            [
                float(mapped_score)
                for completion in self_reflection_completions_flat
                if (mapped_score := completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE)) is not None
            ]
        )
        if (
            not np.isnan(average_self_reflection_score)
            and average_self_reflection_score < defaults.SELF_REFLECTION_EXPLAINABILITY_THRESHOLD
        ):
            self_reflection_explanation = _get_lowest_scoring_reflection_explanation(
                self_reflection_completions[best_answer_idx],
            )
            if self_reflection_explanation is not None and len(self_reflection_explanation) > 0:
                explainability_message += self_reflection_explanation
                explainability_message += _add_punctuation_if_necessary(explainability_message) + "\n"
            else:
                explainability_message += "Cannot verify that this response is correct.\n"

        if (
            not np.isnan(average_consistency_score)
            and average_consistency_score < defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD
        ):
            observed_consistency_explanation = _get_observed_consistency_explanation(
                observed_consistency_completions,
                consistency_scores_flat,
                best_answer,
            )
            if observed_consistency_explanation is not None and len(observed_consistency_explanation) > 0:
                explainability_message += observed_consistency_explanation
                explainability_message += _add_punctuation_if_necessary(explainability_message) + "\n"

        if (
            len(explainability_message) < 5
        ):  # the explainability score is low but neither self_reflection or observed_consistency contribute to this issue or we parsed out all relevant text (there are less than 5 characters left).
            explainability_message = "The prompt/response appear atypical or vague."

        cleaned_explainability_message = explainability_message.strip()
    else:
        cleaned_explainability_message = "Did not find a reason to doubt trustworthiness."

    return cleaned_explainability_message


def _get_lowest_scoring_reflection_explanation(
    self_reflection_completions: list[Completion],
) -> str | None:
    min_score_idx, min_score = None, None
    for idx, completion in enumerate(self_reflection_completions):
        if (mapped_score := completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE)) is not None:
            if min_score is None or mapped_score < min_score:
                min_score_idx = idx
                min_score = mapped_score

    if min_score_idx is None:
        return None

    return self_reflection_completions[min_score_idx].explanation


def _add_punctuation_if_necessary(message: str) -> str:
    """Returns a period if the message does not end with punctuation, otherwise returns a space."""

    if any(
        message.strip(" \n").endswith(punct) for punct in [".", "!", "?", ";", ":"]
    ):  # stripping whitespace left over from any parse of the message before checking for punctuation
        return " "
    else:
        return ". "


def _get_observed_consistency_explanation(
    observed_consistency_completions: list[Completion],
    consistency_scores: npt.NDArray[np.float64],
    best_answer: str,
) -> str | None:
    """Returns the parsed answer from the observed consistency completion with the lowest NLI score.
    If there are no valid observed-consistency completions, returns an empty string.

    If more than one reference answer was generated, we only consider non_contradiction scores for the reference answer that was selected.
    """
    min_score_idx = None
    min_score_consistency_answer = None
    for idx, completion in enumerate(observed_consistency_completions):
        consistency_answer = completion.response_fields.get(ExtractedResponseField.ANSWER)
        if consistency_answer is None or consistency_answer == best_answer:
            continue
        if min_score_idx is None or consistency_scores[idx] < consistency_scores[min_score_idx]:
            min_score_idx = idx
            min_score_consistency_answer = consistency_answer

    if min_score_idx is None or min_score_consistency_answer is None:
        return None

    return OBSERVED_CONSISTENCY_EXPLANATION_TEMPLATE.format(
        observed_consistency_completion=min_score_consistency_answer
    )
