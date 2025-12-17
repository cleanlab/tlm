from tlm.templates.reference_completion_template import ReferenceCompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField
from tlm.config.presets import ReasoningEffort

import pytest

from tests.helpers.litellm_patches import patch_acompletion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, llm_response",
    [
        ("Paris", "Response: [Paris]"),
        ("Paris", "Response: Paris"),
        ("Paris", "Response: [Paris]"),
        ("The capital of France is Paris.", "The capital of France is Paris."),
    ],
)
async def test_generate_completion_from_reference_template(
    reference_template: ReferenceCompletionTemplate,
    answer: str,
    llm_response: str,
) -> None:
    with patch_acompletion(llm_response):
        completion = await generate_completion(
            reference_template,
            template_kwargs={"prompt": "What is the capital of France?"},
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is None
    assert completion.explanation is None
    assert completion.response_fields.get(ExtractedResponseField.ANSWER) == answer
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None


@pytest.mark.asyncio
async def test_generate_completion_from_reference_template_with_logprobs(
    reference_template: ReferenceCompletionTemplate,
) -> None:
    answer = "Paris"
    llm_response = f"Response: {answer}"
    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            reference_template,
            template_kwargs={"prompt": "What is the capital of France?"},
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.explanation is None
    assert completion.response_fields.get(ExtractedResponseField.ANSWER) == answer
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None
    assert completion.perplexity is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, explanation, llm_response",
    [
        (
            "Paris",
            "France is a country in Europe, and Paris has been its capital city for many centuries",
            "Reasoning: [France is a country in Europe, and Paris has been its capital city for many centuries], Response: [Paris]",
        ),
        (
            "Paris",
            "France is a country in Europe, and Paris has been its capital city for many centuries",
            "Reasoning: France is a country in Europe, and Paris has been its capital city for many centuries \nResponse: Paris",
        ),
        ("Paris", None, "Response: [Paris]"),
        ("Paris", None, "Response: Paris"),
    ],
)
async def test_generate_completion_from_reference_template_with_reasoning(
    reference_template_with_reasoning: ReferenceCompletionTemplate,
    answer: str,
    explanation: str | None,
    llm_response: str,
) -> None:
    with patch_acompletion(llm_response):
        completion = await generate_completion(
            reference_template_with_reasoning,
            template_kwargs={"prompt": "What is the capital of France?", "max_explanation_words": 200},
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is None
    if explanation is not None:
        assert completion.explanation == explanation
        assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    else:
        assert completion.explanation is None
        assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None
    assert completion.response_fields.get(ExtractedResponseField.ANSWER) == answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_response, expected_constrained_output",
    [
        ("Response: True", "True"),
        ("Response: true", "True"),
        ("Response: False", "False"),
        ("Response: Invalid text should default to last option", "False"),
    ],
)
async def test_generate_reference_completion_constrain_outputs(
    llm_response: str,
    expected_constrained_output: str,
) -> None:
    template = ReferenceCompletionTemplate.create(
        reasoning_effort=ReasoningEffort.NONE, constrain_outputs=["True", "False"]
    )

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={"prompt": "Answer true or false: Is the sky blue?"},
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.ANSWER) == expected_constrained_output
