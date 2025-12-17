# from tlm.templates.reference_completion_template import ReferenceCompletionTemplate
# from tlm.utils.completion_utils import generate_completion
# from tlm.types import Completion, ParsedResponseField
# from tlm.config.models import CLAUDE_3_5_HAIKU, GPT_4_1_MINI, GPT_4O, O3, O4_MINI, CLAUDE_3_7_SONNET

# import pytest

"""
VCR tests are commented out because they don't work in CI
Retaining in case we want to fix them in the future
"""


# @pytest.mark.asyncio
# @pytest.mark.parametrize("model", [GPT_4_1_MINI, O4_MINI, CLAUDE_3_7_SONNET])
# async def test_generate_completion_from_reference_template(
#     reference_template: ReferenceCompletionTemplate,
#     model: str,
#     vcr_cassette_async: None,
# ) -> None:
#     completion = await generate_completion(
#         reference_template, template_kwargs={"prompt": "What is the capital of France?"}, model=model
#     )

#     assert isinstance(completion, Completion)
#     assert completion.logprobs is None
#     assert completion.explanation is None
#     assert completion.response_fields.get(ParsedResponseField.ANSWER) is not None
#     assert completion.response_fields.get(ParsedResponseField.EXPLANATION) is None


# @pytest.mark.asyncio
# @pytest.mark.parametrize("model", [GPT_4O, O3, CLAUDE_3_5_HAIKU])
# async def test_generate_completion_from_reference_template_with_reasoning(
#     reference_template_with_reasoning: ReferenceCompletionTemplate,
#     model: str,
#     vcr_cassette_async: None,
# ) -> None:
#     completion = await generate_completion(
#         reference_template_with_reasoning,
#         template_kwargs={"prompt": "What is the capital of France?", "max_explanation_words": 200},
#         model=model,
#     )

#     assert isinstance(completion, Completion)
#     assert completion.logprobs is None
#     assert completion.explanation is not None
#     assert completion.response_fields.get(ParsedResponseField.ANSWER) is not None
#     assert completion.response_fields.get(ParsedResponseField.EXPLANATION) is not None


# @pytest.mark.vcr()
# @pytest.mark.asyncio
# async def test_generate_completion_structured_output(
#     reference_template_with_reasoning: ReferenceCompletionTemplate,
# ) -> None:
#     completion = await generate_completion(reference_template_with_reasoning, template_kwargs={"prompt": "What is the capital of France?"})
