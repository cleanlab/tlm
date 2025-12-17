import pytest
import json
from typing import Any

from tlm.components.completions.observed_consistency_completion_generator import ObservedConsistencyCompletionGenerator
from tlm.config.presets import ReasoningEffort
from tlm.types import ExtractedResponseField

from tests.helpers.litellm_patches import patch_acompletion
from tests.helpers.parse_helpers import parse_dict_string


@pytest.mark.asyncio
async def test_basic_consistency_completions() -> None:
    component = ObservedConsistencyCompletionGenerator(
        completion_params={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                }
            ]
        },
        count=1,
        temperature=0.0,
        reasoning_effort=ReasoningEffort.NONE,
        constrain_outputs=None,
    )
    with patch_acompletion("Paris"):
        await component.execute()

    consistency_answers = component.execution_context.get("consistency_answers")
    assert isinstance(consistency_answers, list)
    assert len(consistency_answers) == 1
    assert consistency_answers[0] == "Paris"

    consistency_completions = component.execution_context.get("consistency_completions")
    assert isinstance(consistency_completions, list)
    assert len(consistency_completions) == 1
    assert consistency_completions[0].response_fields[ExtractedResponseField.ANSWER] == "Paris"


@pytest.mark.asyncio
async def test_basic_consistency_completions_with_reasoning() -> None:
    component = ObservedConsistencyCompletionGenerator(
        completion_params={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                }
            ]
        },
        count=1,
        temperature=0.0,
        reasoning_effort=ReasoningEffort.MEDIUM,
        constrain_outputs=None,
    )
    with patch_acompletion("<think>\nThe capital of France is Paris.\n</think><answer>\nParis\n</answer>"):
        await component.execute()

    consistency_answers = component.execution_context.get("consistency_answers")
    assert isinstance(consistency_answers, list)
    assert len(consistency_answers) == 1
    assert consistency_answers[0] == "Paris"

    consistency_completions = component.execution_context.get("consistency_completions")
    assert isinstance(consistency_completions, list)
    assert len(consistency_completions) == 1
    assert (
        consistency_completions[0].response_fields[ExtractedResponseField.EXPLANATION]
        == "The capital of France is Paris."
    )
    assert consistency_completions[0].response_fields[ExtractedResponseField.ANSWER] == "Paris"


@pytest.mark.asyncio
async def test_structured_outputs_consistency_completions(
    structured_outputs_completion_params: dict[str, Any],
) -> None:
    component = ObservedConsistencyCompletionGenerator(
        count=1,
        temperature=0.0,
        reasoning_effort=ReasoningEffort.MEDIUM,
        constrain_outputs=None,
        completion_params=structured_outputs_completion_params,
    )

    answer_dict = {
        "name": "Science Fair",
        "date": "Friday",
        "participants": ["Alice", "Bob"],
    }
    response_dict = {
        "explanation": (
            "The event is a science fair scheduled for Friday. "
            "Participants mentioned are Alice and Bob. "
            "No specific date given, so we use the weekday provided."
        ),
        "answer": answer_dict,
    }

    with patch_acompletion(json.dumps(response_dict)):
        await component.execute()

    consistency_answers = component.execution_context.get("consistency_answers")
    assert isinstance(consistency_answers, list)
    assert len(consistency_answers) == 1
    assert parse_dict_string(consistency_answers[0]) == answer_dict

    consistency_completions = component.execution_context.get("consistency_completions")
    assert isinstance(consistency_completions, list)
    assert len(consistency_completions) == 1
    assert parse_dict_string(consistency_completions[0].response_fields[ExtractedResponseField.ANSWER]) == answer_dict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, llm_response, expected_constrained_output, explanation",
    [
        (
            "The Earth is a beautiful place.",
            "<think>\nThe text conveys an admiration and appreciation for the Earth.\n</think><answer>\npositive\n</answer>",
            "positive",
            "The text conveys an admiration and appreciation for the Earth.",
        ),
        (
            "The Earth is a terrible place.",
            "<think>\nThe text conveys a sense of disgust or disapproval of the Earth.\n</think><answer>\nnegative\n</answer>",
            "negative",
            "The text conveys a sense of disgust or disapproval of the Earth.",
        ),
        (
            "The Earth is a fine place.",
            "<think>\nThe text is neutral and does not convey a strong positive or negative sentiment towards the Earth.\n</think><answer>\nneutral\n</answer>",
            "neutral",
            "The text is neutral and does not convey a strong positive or negative sentiment towards the Earth.",
        ),
    ],
)
async def test_qa_consistency_completions_constrain_outputs(
    text: str,
    llm_response: str,
    expected_constrained_output: str,
    explanation: str,
) -> None:
    component = ObservedConsistencyCompletionGenerator(
        completion_params={
            "messages": [
                {
                    "role": "user",
                    "content": f"Classify the tone of the Text as positive, negative, or neutral.\nText: {text}",
                }
            ]
        },
        count=1,
        temperature=0.0,
        reasoning_effort=ReasoningEffort.MEDIUM,
        constrain_outputs=["positive", "negative", "neutral"],
    )

    with patch_acompletion(llm_response, logprobs=True):
        await component.execute()

    consistency_answers = component.execution_context.get("consistency_answers")
    assert isinstance(consistency_answers, list)
    assert len(consistency_answers) == 1
    assert consistency_answers[0] == expected_constrained_output

    consistency_completions = component.execution_context.get("consistency_completions")
    assert isinstance(consistency_completions, list)
    assert len(consistency_completions) == 1
    assert consistency_completions[0].response_fields[ExtractedResponseField.ANSWER] == expected_constrained_output
    assert consistency_completions[0].explanation == explanation
    assert consistency_completions[0].perplexity is not None
