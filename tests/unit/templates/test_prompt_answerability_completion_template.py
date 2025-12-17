from tlm.templates.prompt_evaluation_completion_template import PromptAnswerabilityCompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField
from tlm.utils.parse_utils import get_choice_token_confidence

import pytest

from tests.helpers.litellm_patches import patch_acompletion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice, expected_mapped_score",
    [
        ("Yes", 1.0),
        ("No", 0.0),
    ],
)
async def test_prompt_answerability_completion_template_choices(
    choice: str,
    expected_mapped_score: float,
) -> None:
    """Test PromptAnswerabilityCompletionTemplate with Yes/No choices."""
    template = PromptAnswerabilityCompletionTemplate.create()

    llm_response = f"<choice>\n{choice}\n</choice>"

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "prompt": "What is the capital of France?",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == choice
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == expected_mapped_score
    scored_answer_token_confidence = get_choice_token_confidence(completion)
    assert scored_answer_token_confidence is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "What is the capital of France?",
        "How much water does your Simple Water Bottle hold?",
        "Answer the user Question using the Context.\nQuestion: How much water does your Simple Water Bottle hold?\nContext: The Simple Water Bottle is a reusable 27 oz water bottle.",
        "Hello, how are you?",
    ],
)
async def test_prompt_answerability_completion_template_with_different_prompts(
    prompt: str,
) -> None:
    """Test PromptAnswerabilityCompletionTemplate with different prompt formats."""
    template = PromptAnswerabilityCompletionTemplate.create()

    choice = "Yes"
    llm_response = f"<choice>\n{choice}\n</choice>"

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "prompt": prompt,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == choice
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 1.0
    scored_answer_token_confidence = get_choice_token_confidence(completion)
    assert scored_answer_token_confidence is not None


@pytest.mark.asyncio
async def test_prompt_answerability_completion_template_no_choice() -> None:
    """Test PromptAnswerabilityCompletionTemplate when choice is No."""
    template = PromptAnswerabilityCompletionTemplate.create()

    choice = "No"
    llm_response = f"<choice>\n{choice}\n</choice>"

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "prompt": "What is the answer to this impossible question that requires information not available?",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == choice
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.0
    scored_answer_token_confidence = get_choice_token_confidence(completion)
    assert scored_answer_token_confidence is not None
