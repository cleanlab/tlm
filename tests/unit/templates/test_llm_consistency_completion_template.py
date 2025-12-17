import pytest

from tlm.templates.llm_consistency_completion_templates import (
    CodeConsistencyCompletionTemplate,
    StatementConsistencyCompletionTemplate,
)
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField

from tests.helpers.litellm_patches import patch_acompletion


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_choice", ["A", "B", "a", "b"])
async def test_code_consistency_completion_template(llm_choice: str) -> None:
    template = CodeConsistencyCompletionTemplate.create()

    llm_response = f"Choice: [{llm_choice}]"

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "input_1": "print('Hello, world!')",
                "input_2": "fruits = ['apple', 'banana', 'cherry']",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == llm_choice
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == (
        1.0 if llm_choice.lower() == "a" else 0.0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_choice", ["Yes", "No", "yes", "no"])
async def test_statement_consistency_completion_template(llm_choice: str) -> None:
    template = StatementConsistencyCompletionTemplate.create()

    llm_response = f"<answer>\n{llm_choice}\n</answer>"

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "input_1": "The sky is blue.",
                "input_2": "The sky is red.",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == llm_choice
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == (
        1.0 if llm_choice.lower() == "yes" else 0.0
    )
