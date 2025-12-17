import json
import pytest

from tlm.config.presets import ReasoningEffort
from tlm.templates.reflection_completion_templates import (
    ReflectionKnowledgeGapTemplate,
    ReflectionBinaryCorrectnessTemplate,
    ReflectionArgumentTemplate,
    ReflectionTrustworthinessTemplate,
    ReflectionCorrectnessTemplate,
    ReflectionRAGCertaintyTemplate,
    ReflectionSOPerScoreCorrectnessTemplate,
    ReflectionSOPerScoreCertaintyTemplate,
    ReflectionCertaintyTemplate,
    ReflectionRAGArgumentTemplate,
    ReflectionRAGIssuesTemplate,
    ReflectionClassificationCorrectnessTemplate,
    ReflectionClassificationScoringTemplate,
)
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField
from tlm.utils.parse_utils import get_choice_token_confidence

from tests.helpers.litellm_patches import patch_acompletion


@pytest.mark.asyncio
async def test_reflection_knowledge_gap_template() -> None:
    template = ReflectionKnowledgeGapTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    rating = "10"
    explanation = "Let's think step by step. Spelling out the word Strawberry: S-T-R-A-W-B-E-R-R-Y. There are 2 Rs in Strawberry. Therefore the Response is correct."
    llm_response = f"""<think>
{explanation}
</think>

<rating>
{rating}
</rating>"""

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How many Rs in strawberry?",
                "answer": "2",
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 1.0


@pytest.mark.asyncio
async def test_reflection_argument_template() -> None:
    template = ReflectionArgumentTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    score = "0"
    explanation = "Spelling out the word Strawberry: S-T-R-A-W-B-E-R-R-Y. There are 3 Rs in Strawberry. Therefore the Response is wrong."
    llm_response = f"""<think>
{explanation}
</think>

<score>
{score}
</score>"""

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How many Rs in strawberry?",
                "answer": "2",
                "max_explanation_words": 200,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is None
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == score
    assert completion.perplexity is None
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.0


@pytest.mark.asyncio
async def test_reflection_trustworthiness_template() -> None:
    template = ReflectionTrustworthinessTemplate.create(reasoning_effort=ReasoningEffort.HIGH)

    score = "3"
    explanation = "The United States has a population of 340 million people as of 2024. The answer is close to the true value, but not exactly correct."

    llm_response = f"""<think>
{explanation}
</think>

<rating>
{score}
</rating>"""

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How many people live in the United States?",
                "answer": "320 million",
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == score
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.5


@pytest.mark.asyncio
async def test_reflection_correctness_template() -> None:
    template = ReflectionCorrectnessTemplate.create(reasoning_effort=ReasoningEffort.NONE)

    rating = "10"
    llm_response = f"<rating>\n{rating}\n</rating>"

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How many people live in the United States?",
                "answer": "320 million",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 1.0


@pytest.mark.asyncio
async def test_reflection_rag_certainty_template() -> None:
    template = ReflectionRAGCertaintyTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    score = "10"
    explanation = "The United States has a population of 340 million people as of 2024. The answer is close to the true value, but not exactly correct."
    llm_response = f"""<think>
{explanation}
</think>

<score>
{score}
</score>"""

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How many people live in the United States?",
                "answer": "320 million",
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == score
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 10 / 100.0


@pytest.mark.asyncio
async def test_reflection_binary_correctness_template() -> None:
    template = ReflectionBinaryCorrectnessTemplate.create(reasoning_effort=ReasoningEffort.NONE)

    choice = "True"
    llm_response = f"<choice>\n{choice}\n</choice>"

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How many Rs in strawberry?",
                "answer": "2",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == choice
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 1.0

    scored_answer_token_confidence = get_choice_token_confidence(completion)
    assert scored_answer_token_confidence is not None


@pytest.mark.asyncio
async def test_reflection_so_per_score_correctness_template() -> None:
    template = ReflectionSOPerScoreCorrectnessTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    answer_dict = {
        "name": "Science Fair",
        "date": "Friday",
        "participants": ["Alice", "Bob"],
    }
    answer_json = json.dumps(answer_dict)

    llm_response_dict = {
        "name": {
            "explanation": "The name 'Science Fair' accurately matches the event described in the request.",
            "confidence": "Certain",
        },
        "date": {
            "explanation": "The date 'Friday' is mentioned in the request, but the specific Friday is not clear without context.",
            "confidence": "Mostly Certain",
        },
        "participants": {
            "explanation": "The participants list correctly includes Alice and Bob as mentioned in the request.",
            "confidence": "Certain",
        },
    }
    llm_response = json.dumps(llm_response_dict)

    response_format_model = template.construct_response_format(answer_json)

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "Extract the event information.",
                "answer": answer_json,
                "max_explanation_words": 125,
            },
            response_format_model=response_format_model,
        )

    assert isinstance(completion, Completion)

    metadata_per_field = completion.per_field_metadata
    assert metadata_per_field is not None
    assert "name" in metadata_per_field
    assert "date" in metadata_per_field
    assert "participants" in metadata_per_field

    assert metadata_per_field["name"].score == 1.0
    assert metadata_per_field["date"].score == 0.75
    assert metadata_per_field["participants"].score == 1.0

    assert metadata_per_field["name"].explanation == llm_response_dict["name"]["explanation"]
    assert metadata_per_field["date"].explanation == llm_response_dict["date"]["explanation"]
    assert metadata_per_field["participants"].explanation == llm_response_dict["participants"]["explanation"]

    # Harmonic mean of [1.0, 0.75, 1.0] ≈ 0.901
    mapped_score = completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE)
    assert mapped_score is not None
    assert 0.89 < mapped_score < 0.91  # Allow small floating point differences


@pytest.mark.asyncio
async def test_reflection_so_per_score_certainty_template() -> None:
    template = ReflectionSOPerScoreCertaintyTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    answer_dict = {
        "name": "Science Fair",
        "date": "Friday",
        "participants": ["Alice", "Bob"],
    }
    answer_json = json.dumps(answer_dict)

    llm_response_dict = {
        "name": {
            "explanation": "The name 'Science Fair' accurately matches the event described in the request.",
            "score": 100,
        },
        "date": {
            "explanation": "The date 'Friday' is mentioned in the request, but the specific Friday is not clear without context.",
            "score": 75,
        },
        "participants": {
            "explanation": "The participants list correctly includes Alice and Bob as mentioned in the request.",
            "score": 100,
        },
    }
    llm_response = json.dumps(llm_response_dict)

    response_format_model = template.construct_response_format(answer_json)

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "Extract the event information.",
                "answer": answer_json,
                "max_explanation_words": 125,
            },
            response_format_model=response_format_model,
        )

    assert isinstance(completion, Completion)

    metadata_per_field = completion.per_field_metadata
    assert metadata_per_field is not None
    assert "name" in metadata_per_field
    assert "date" in metadata_per_field
    assert "participants" in metadata_per_field

    assert metadata_per_field["name"].score == 1.0
    assert metadata_per_field["date"].score == 0.75
    assert metadata_per_field["participants"].score == 1.0

    assert metadata_per_field["name"].explanation == llm_response_dict["name"]["explanation"]
    assert metadata_per_field["date"].explanation == llm_response_dict["date"]["explanation"]
    assert metadata_per_field["participants"].explanation == llm_response_dict["participants"]["explanation"]

    # Harmonic mean of [1.0, 0.75, 1.0] ≈ 0.901
    mapped_score = completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE)
    assert mapped_score is not None
    assert 0.89 < mapped_score < 0.91  # Allow small floating point differences


@pytest.mark.asyncio
async def test_reflection_certainty_template() -> None:
    template = ReflectionCertaintyTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    score = "85"
    explanation = (
        "The answer is factually correct based on available information. The capital of France is indeed Paris."
    )
    llm_response = f"""<think>
{explanation}
</think>

<score>
{score}
</score>"""

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "What is the capital of France?",
                "answer": "Paris",
                "max_explanation_words": 125,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == score
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 85 / 100.0


@pytest.mark.asyncio
async def test_reflection_rag_argument_template() -> None:
    template = ReflectionRAGArgumentTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    score = "8"
    explanation = (
        "The response accurately reflects the information provided in the context about the water bottle capacity."
    )
    llm_response = f"""<think>
{explanation}
</think>

<score>
{score}
</score>"""

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How much water does your Simple Water Bottle hold?",
                "answer": "27 oz",
                "max_explanation_words": 125,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == score
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 8 / 10.0


@pytest.mark.asyncio
async def test_reflection_rag_issues_template() -> None:
    template = ReflectionRAGIssuesTemplate.create(reasoning_effort=ReasoningEffort.MEDIUM)

    score = "70"
    issues = "- The response might be missing unit clarification\n- Could specify if this is US fluid ounces"
    alternate_response = "The Simple Water Bottle holds 27 US fluid ounces of water."
    llm_response = f"""<issues>
{issues}
</issues>

<alternate_response>
{alternate_response}
</alternate_response>

<score>
{score}
</score>"""

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "How much water does your Simple Water Bottle hold?",
                "answer": "27 oz",
                "max_explanation_words": 125,
            },
        )

    assert isinstance(completion, Completion)
    assert (
        completion.response_fields.get(ExtractedResponseField.EXPLANATION) == issues[2:]
    )  # parser strips the leading non-word chars
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == score
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 70 / 100.0


@pytest.mark.asyncio
async def test_reflection_classification_correctness_template() -> None:
    template = ReflectionClassificationCorrectnessTemplate.create(reasoning_effort=ReasoningEffort.NONE)

    choice = "A"
    llm_response = f"Choice: {choice}"

    with patch_acompletion(llm_response, logprobs=True):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "Is Python a programming language?",
                "answer": "Yes",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.logprobs is not None
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == choice
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 1.0

    scored_answer_token_confidence = get_choice_token_confidence(completion)
    assert scored_answer_token_confidence is not None


@pytest.mark.asyncio
async def test_reflection_classification_scoring_template() -> None:
    template = ReflectionClassificationScoringTemplate.create(reasoning_effort=ReasoningEffort.NONE)

    rating = "4"
    llm_response = f"Rating: {rating}"

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "question": "Is Python a programming language?",
                "answer": "Yes",
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.75  # 4 maps to 0.75
