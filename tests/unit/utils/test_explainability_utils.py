import numpy as np

from tlm.config.defaults import get_settings
from tlm.utils.explainability_utils import (
    HIGH_CONFIDENCE_MESSAGE,
    FALLBACK_EXPLANATION_MESSAGE,
    NO_SELF_REFLECTION_EXPLANATION_MESSAGE,
    OBSERVED_CONSISTENCY_EXPLANATION_TEMPLATE,
    _add_punctuation_if_necessary,
    _get_lowest_scoring_reflection_explanation,
    _get_observed_consistency_explanation,
    get_explainability_message,
)
from tlm.types import Completion, ExtractedResponseField

defaults = get_settings()


def test_get_explainability_message_no_confidence_score() -> None:
    assert get_explainability_message(None, [], [], 0, np.array([]), 0, "test") == ""


def test_get_explainability_message_low_confidence_no_self_reflection_or_consistency() -> None:
    assert (
        get_explainability_message(defaults.EXPLAINABILITY_THRESHOLD - 0.1, [], [], np.nan, np.array([]), 0, "test")
        == FALLBACK_EXPLANATION_MESSAGE
    )


def test_get_explainability_message_low_confidence_with_self_reflection_explanation() -> None:
    self_reflection_explanation = "Self reflection explanation"
    self_reflection_completion = Completion(
        message="test",
        explanation=self_reflection_explanation,
        response_fields={ExtractedResponseField.MAPPED_SCORE: defaults.SELF_REFLECTION_EXPLAINABILITY_THRESHOLD - 0.1},
        original_response={},
        template=None,
    )
    assert self_reflection_explanation in get_explainability_message(
        defaults.EXPLAINABILITY_THRESHOLD - 0.1,
        [[self_reflection_completion]],
        [],
        np.nan,
        np.array([]),
        0,
        "test",
    )


def test_get_explainability_message_low_confidence_with_self_reflection_no_explanation() -> None:
    self_reflection_completion = Completion(
        message="test",
        explanation=None,
        response_fields={ExtractedResponseField.MAPPED_SCORE: defaults.SELF_REFLECTION_EXPLAINABILITY_THRESHOLD - 0.1},
        original_response={},
        template=None,
    )
    assert (
        get_explainability_message(
            defaults.EXPLAINABILITY_THRESHOLD - 0.1,
            [[self_reflection_completion]],
            [],
            np.nan,
            np.array([]),
            0,
            "test",
        )
        == NO_SELF_REFLECTION_EXPLANATION_MESSAGE.strip()
    )


def test_get_explainability_message_high_confidence_score() -> None:
    assert get_explainability_message(0.9, [], [], 0, np.array([]), 0, "test") == HIGH_CONFIDENCE_MESSAGE


def test_get_explainaibility_message_nan_confidence_score() -> None:
    assert get_explainability_message(np.nan, [], [], 0, np.array([]), 0, "test") == HIGH_CONFIDENCE_MESSAGE


def test_get_explainability_message_low_consistency_score() -> None:
    observed_consistency_answer = "incorrect answer"
    observed_consistency_completion = Completion(
        message="test",
        response_fields={
            ExtractedResponseField.MAPPED_SCORE: defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD - 0.1,
            ExtractedResponseField.ANSWER: observed_consistency_answer,
        },
        original_response={},
        template=None,
    )
    assert OBSERVED_CONSISTENCY_EXPLANATION_TEMPLATE.format(
        observed_consistency_completion=observed_consistency_answer
    ) in get_explainability_message(
        defaults.EXPLAINABILITY_THRESHOLD - 0.1,
        [],
        [observed_consistency_completion],
        defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD - 0.1,
        np.array([defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD - 0.1]),
        0,
        "test",
    )


def test_get_explainability_message_self_reflection_and_consistency_explanations() -> None:
    self_reflection_explanation = "Self reflection explanation"
    self_reflection_completion = Completion(
        message="test",
        explanation=self_reflection_explanation,
        response_fields={ExtractedResponseField.MAPPED_SCORE: defaults.SELF_REFLECTION_EXPLAINABILITY_THRESHOLD - 0.1},
        original_response={},
        template=None,
    )
    observed_consistency_answer = "incorrect answer"
    observed_consistency_completion = Completion(
        message="test",
        response_fields={
            ExtractedResponseField.MAPPED_SCORE: defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD - 0.1,
            ExtractedResponseField.ANSWER: observed_consistency_answer,
        },
        original_response={},
        template=None,
    )
    res = get_explainability_message(
        defaults.EXPLAINABILITY_THRESHOLD - 0.1,
        [[self_reflection_completion]],
        [observed_consistency_completion],
        defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD - 0.1,
        np.array([defaults.CONSISTENCY_EXPLAINABILITY_THRESHOLD - 0.1]),
        0,
        "test",
    )

    assert self_reflection_explanation in res
    assert (
        OBSERVED_CONSISTENCY_EXPLANATION_TEMPLATE.format(observed_consistency_completion=observed_consistency_answer)
        in res
    )


def test_get_lowest_scoring_reflection_explanation() -> None:
    self_reflection_completions = [
        Completion(
            message="test",
            explanation="Self reflection explanation",
            response_fields={
                ExtractedResponseField.MAPPED_SCORE: defaults.SELF_REFLECTION_EXPLAINABILITY_THRESHOLD - 0.1
            },
            original_response={},
            template=None,
        ),
        Completion(
            message="test",
            explanation="Self reflection explanation 2",
            response_fields={
                ExtractedResponseField.MAPPED_SCORE: defaults.SELF_REFLECTION_EXPLAINABILITY_THRESHOLD - 0.2
            },
            original_response={},
            template=None,
        ),
    ]
    assert _get_lowest_scoring_reflection_explanation(self_reflection_completions) == "Self reflection explanation 2"


def test_get_lowest_scoring_reflection_explanation_no_explanation() -> None:
    self_reflection_completions = [
        Completion(
            message="test",
            explanation=None,
            response_fields={},
            original_response={},
            template=None,
        )
    ]
    assert _get_lowest_scoring_reflection_explanation(self_reflection_completions) is None


def test_add_punctuation_if_necessary() -> None:
    assert _add_punctuation_if_necessary("Hello, world!") == " "
    assert _add_punctuation_if_necessary("Hello, world!?") == " "
    assert _add_punctuation_if_necessary("Hello, world.") == " "
    assert _add_punctuation_if_necessary("Hello, world") == ". "
    assert _add_punctuation_if_necessary("Hello, world;\n") == " "
    assert _add_punctuation_if_necessary("Hello, world: ") == " "


def test_get_observed_consistency_explanation() -> None:
    answer1 = "Answer 1"
    answer2 = "Answer 2"
    best_answer = "correct answer"
    observed_consistency_completions = [
        Completion(
            message="test",
            explanation=None,
            response_fields={
                ExtractedResponseField.MAPPED_SCORE: 0.3,
                ExtractedResponseField.ANSWER: answer1,
            },
            original_response={},
            template=None,
        ),
        Completion(
            message="test",
            explanation=None,
            response_fields={
                ExtractedResponseField.MAPPED_SCORE: 0.2,
                ExtractedResponseField.ANSWER: answer2,
            },
            original_response={},
            template=None,
        ),
        Completion(
            message="test",
            explanation=None,
            response_fields={ExtractedResponseField.MAPPED_SCORE: 0.1, ExtractedResponseField.ANSWER: best_answer},
            original_response={},
            template=None,
        ),
    ]
    assert _get_observed_consistency_explanation(
        observed_consistency_completions, np.array([0.3, 0.2, 0.1]), best_answer
    ) == OBSERVED_CONSISTENCY_EXPLANATION_TEMPLATE.format(observed_consistency_completion=answer2)


def test_get_observed_consistency_explanation_no_explanation() -> None:
    observed_consistency_completions = [
        Completion(
            message="test",
            explanation=None,
            response_fields={},
            original_response={},
            template=None,
        )
    ]
    assert (
        _get_observed_consistency_explanation(observed_consistency_completions, np.array([]), "correct answer") is None
    )
