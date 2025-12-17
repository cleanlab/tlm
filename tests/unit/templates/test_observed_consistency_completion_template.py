from tlm.config.presets import ReasoningEffort
from tlm.templates.observed_consistency_completion_template import ObservedConsistencyQACompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField

import pytest

from tests.helpers.litellm_patches import patch_acompletion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning_effort, answer, explanation, llm_response",
    [
        (ReasoningEffort.NONE, "Paris", None, "<answer>\nParis\n</answer>"),
        (
            ReasoningEffort.MEDIUM,
            "Paris",
            "France is a country in Europe, and Paris has been its capital city for many centuries.",
            "<think>\nFrance is a country in Europe, and Paris has been its capital city for many centuries.\n</think>\n\n<answer>\nParis\n</answer>",
        ),
        (
            ReasoningEffort.HIGH,
            "27 oz",
            "The context clearly states that the Simple Water Bottle holds 27 oz of water.",
            "<think>\nThe context clearly states that the Simple Water Bottle holds 27 oz of water.\n</think>\n\n<answer>\n27 oz\n</answer>",
        ),
    ],
)
async def test_observed_consistency_qa_completion_template_with_reasoning_effort(
    reasoning_effort: ReasoningEffort,
    answer: str,
    explanation: str | None,
    llm_response: str,
) -> None:
    """Test ObservedConsistencyQACompletionTemplate with different reasoning effort levels."""
    template = ObservedConsistencyQACompletionTemplate.create(reasoning_effort=reasoning_effort)

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "What is the capital of France?",
                "max_explanation_words": 100 if reasoning_effort != ReasoningEffort.NONE else 0,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.ANSWER) == answer

    if explanation is not None:
        assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    else:
        assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None


@pytest.mark.asyncio
async def test_observed_consistency_qa_completion_template_with_constrain_outputs() -> None:
    """Test ObservedConsistencyQACompletionTemplate with constrained outputs."""
    template = ObservedConsistencyQACompletionTemplate.create(
        reasoning_effort=ReasoningEffort.MEDIUM, constrain_outputs=["positive", "negative", "neutral"]
    )

    answer = "positive"
    explanation = "The text conveys an admiration and appreciation for the Earth."
    llm_response = f"""<think>
{explanation}
</think>

<answer>
{answer}
</answer>"""

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "Classify the tone of the Text as positive, negative, or neutral.\nText: The Earth is a beautiful place.",
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.ANSWER) == answer
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.perplexity is not None
