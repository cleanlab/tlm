from typing import Callable

import pytest
from litellm.types.utils import ChoiceLogprobs

from tlm.templates.keywords import PROMPT_PLACEHOLDER
from tlm.templates.parsers import (
    ANSWER_XML_PARSER,
    CHOICE_YES_NO_XML_PARSER,
    REASONING_RESPONSE_PARSER,
    RESPONSE_PARSER,
    SCORE_10_XML_PARSER,
    SCORE_XML_PARSER,
)
from tlm.templates.score_mapping import (
    score_5_mapping,
    score_10_mapping,
    yes_no_mapping,
)
from tlm.types import (
    Completion,
    ExtractedResponseField,
    CompletionTemplate,
    RegexPattern,
    AnswerChoiceToken,
)
from tlm.utils.completion_utils import _parse_completion


@pytest.fixture
def basic_template() -> CompletionTemplate:
    """Basic CompletionTemplate for testing."""
    return CompletionTemplate(prompt_template="Test prompt")


@pytest.fixture
def completion_with_logprobs() -> Completion:
    """Completion object with logprobs."""
    logprobs = ChoiceLogprobs.model_construct(content=[])
    return Completion(
        message="Answer: Yes",
        logprobs=logprobs,
        original_response={"response": "Answer: Yes"},
        template=None,
    )


@pytest.fixture
def completion_without_logprobs() -> Completion:
    """Completion object without logprobs."""
    return Completion(
        message="Answer: Yes",
        logprobs=None,
        original_response={"response": "Answer: Yes"},
        template=None,
    )


@pytest.fixture
def answer_choice_tokens() -> list[AnswerChoiceToken]:
    """AnswerChoiceToken list for testing."""
    return [
        AnswerChoiceToken(token="yes", positive=True),
        AnswerChoiceToken(token="no", positive=False),
    ]


def test_format_prompt() -> None:
    template = CompletionTemplate(prompt_template=f"Here is a User Query: {PROMPT_PLACEHOLDER}")
    formatted_prompt = template.format_messages(prompt="What's the coldest planet?")
    assert formatted_prompt == [{"role": "user", "content": "Here is a User Query: What's the coldest planet?"}]


def test_parse_completion_extracts_fields_from_regex_match() -> None:
    """Test that regex patterns correctly extract response fields."""
    template = CompletionTemplate(
        prompt_template="Test prompt",
        parse_patterns=REASONING_RESPONSE_PARSER,
    )
    completion = Completion(
        message="Reasoning: [It is the capital of France], Response: [Paris]",
        original_response={"response": "Reasoning: [It is the capital of France], Response: [Paris]"},
        template=template,
    )

    _parse_completion(completion)

    assert completion.response_fields[ExtractedResponseField.ANSWER] == "Paris"
    assert completion.response_fields[ExtractedResponseField.EXPLANATION] == "It is the capital of France"


def test_parse_completion_matches_all_fields() -> None:
    """Test that parsing stops after the first matching pattern."""
    template = CompletionTemplate(
        prompt_template="Test prompt",
        parse_patterns=RESPONSE_PARSER | SCORE_XML_PARSER,
    )
    completion = Completion(
        message="Response: [Paris] <score>5</score>",
        original_response={"response": "Response: [Paris] <score>5</score>"},
        template=template,
    )

    _parse_completion(completion)

    assert completion.response_fields[ExtractedResponseField.ANSWER] == "Paris"
    assert completion.response_fields[ExtractedResponseField.SCORE] == "5"


def test_parse_completion_with_answer_field_calls_constrain_output() -> None:
    """Test that constrain_output is called when ANSWER field is parsed and constrain_outputs is set."""
    template = CompletionTemplate(
        prompt_template="Test prompt",
        parse_patterns=ANSWER_XML_PARSER,
        constrain_outputs=["Paris", "London"],
    )
    completion = Completion(
        message="<answer>paris</answer>",
        original_response={"response": "<answer>paris</answer>"},
        template=template,
    )

    _parse_completion(completion)

    assert completion.response_fields[ExtractedResponseField.ANSWER] == "Paris"


@pytest.mark.parametrize(
    "parsers,message,field,expected_field_value,expected_mapped_score,score_mapper",
    [
        (
            SCORE_XML_PARSER,
            "<score>5</score>",
            ExtractedResponseField.SCORE,
            "5",
            1.0,
            score_5_mapping,
        ),
        (
            SCORE_10_XML_PARSER,
            "<score>8</score>",
            ExtractedResponseField.SCORE,
            "8",
            0.8,
            score_10_mapping,
        ),
        (
            CHOICE_YES_NO_XML_PARSER,
            "<choice>Yes</choice>",
            ExtractedResponseField.SCORE,
            "Yes",
            1.0,
            yes_no_mapping,
        ),
    ],
)
def test_parse_completion_applies_score_mapper_to_fields(
    parsers: dict[ExtractedResponseField, list[RegexPattern]],
    message: str,
    field: ExtractedResponseField,
    expected_field_value: str,
    expected_mapped_score: float,
    score_mapper: Callable[[str], float],
) -> None:
    """Test that score_mapper is applied to SCORE, RATING, and CHOICE fields."""
    template = CompletionTemplate(
        prompt_template="Test prompt",
        parse_patterns=parsers,
        score_mapper=score_mapper,
    )
    completion = Completion(
        message=message,
        original_response={"response": message},
        template=template,
    )

    _parse_completion(completion)

    assert completion.response_fields[field] == expected_field_value
    assert completion.response_fields[ExtractedResponseField.MAPPED_SCORE] == expected_mapped_score


@pytest.mark.parametrize(
    "parsers,message,expected_answer",
    [
        (
            RESPONSE_PARSER,
            "Response: [Paris]",
            "Paris",
        ),
        (
            RESPONSE_PARSER,
            "[Paris]",
            "Paris",
        ),
        (
            ANSWER_XML_PARSER,
            "<answer>Paris</answer>",
            "Paris",
        ),
    ],
)
def test_parse_completion_with_multiple_regex_patterns_tries_all(
    parsers: dict[ExtractedResponseField, list[RegexPattern]],
    message: str,
    expected_answer: str,
) -> None:
    """Test that all regex patterns in a ParsePattern are tried."""
    template = CompletionTemplate(
        prompt_template="Test prompt",
        parse_patterns=parsers,
    )
    completion = Completion(
        message=message,
        original_response={"response": message},
        template=template,
    )

    _parse_completion(completion)

    assert completion.response_fields[ExtractedResponseField.ANSWER] == expected_answer


def test_parse_completion_strips_whitespace_from_field_values() -> None:
    """Test that field values are stripped of whitespace."""
    template = CompletionTemplate(
        prompt_template="Test prompt",
        parse_patterns=ANSWER_XML_PARSER,
    )
    completion = Completion(
        message="<answer>   Paris   </answer>",
        original_response={"response": "<answer>   Paris   </answer>"},
        template=template,
    )

    _parse_completion(completion)

    assert completion.response_fields[ExtractedResponseField.ANSWER] == "Paris"
