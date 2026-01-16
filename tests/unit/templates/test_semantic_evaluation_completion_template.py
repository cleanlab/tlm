from tlm.config.presets import ReasoningEffort
from tlm.templates.semantic_evaluation_completion_template import SemanticEvaluationCompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.types import Completion, ExtractedResponseField, Eval

import pytest

from tests.helpers.litellm_patches import patch_acompletion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning_effort, rating, explanation, expected_mapped_score",
    [
        (ReasoningEffort.NONE, "5", None, 1.0),
        (ReasoningEffort.NONE, "4", None, 0.75),
        (ReasoningEffort.NONE, "3", None, 0.5),
        (ReasoningEffort.NONE, "2", None, 0.25),
        (ReasoningEffort.NONE, "1", None, 0.0),
        (
            ReasoningEffort.MEDIUM,
            "5",
            "The context provides complete information about the water bottle capacity.",
            1.0,
        ),
        (
            ReasoningEffort.MEDIUM,
            "3",
            "The context provides partial information but lacks some details.",
            0.5,
        ),
        (
            ReasoningEffort.HIGH,
            "4",
            "The context contains most of the necessary information to answer the question, with only minor gaps.",
            0.75,
        ),
    ],
)
async def test_semantic_evaluation_completion_template_with_reasoning_effort(
    reasoning_effort: ReasoningEffort,
    rating: str,
    explanation: str | None,
    expected_mapped_score: float,
) -> None:
    """Test SemanticEvaluationCompletionTemplate with different reasoning effort levels."""
    eval = Eval(
        name="context_sufficiency",
        criteria="Determine if the Document contains 100% of the information needed to answer the Question.",
        query_identifier="Question",
        context_identifier="Document",
        response_identifier=None,
    )

    template = SemanticEvaluationCompletionTemplate.create(eval=eval, reasoning_effort=reasoning_effort)

    if explanation is not None:
        llm_response = f"""<think>
{explanation}
</think>

<rating>
{rating}
</rating>"""
    else:
        llm_response = f"<rating>\n{rating}\n</rating>"

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "query_identifier": eval.query_identifier,
                "context_identifier": eval.context_identifier,
                "response_identifier": eval.response_identifier,
                "query": "How much water does your Simple Water Bottle hold?",
                "context": "The Simple Water Bottle is a reusable 27 oz water bottle.",
                "reference_answer": None,
                "eval_criteria": eval.criteria,
                "max_explanation_words": 100 if reasoning_effort != ReasoningEffort.NONE else 0,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == expected_mapped_score

    if explanation is not None:
        assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    else:
        assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_identifier, context_identifier, response_identifier, query, context, reference_answer",
    [
        ("Question", "Document", None, "What is the capital?", "Paris is the capital.", None),
        ("Query", "Context", "Response", "How much water?", "27 oz", "The bottle holds 27 oz"),
        ("User Query", None, "AI Assistant Response", "What is Python?", None, "Python is a programming language"),
        ("User Request", None, None, "Hello", None, None),
    ],
)
async def test_semantic_evaluation_completion_template_with_different_identifiers(
    query_identifier: str | None,
    context_identifier: str | None,
    response_identifier: str | None,
    query: str,
    context: str | None,
    reference_answer: str | None,
) -> None:
    """Test SemanticEvaluationCompletionTemplate with different identifier configurations."""
    eval = Eval(
        name="test_eval",
        criteria="Test criteria for evaluation.",
        query_identifier=query_identifier,
        context_identifier=context_identifier,
        response_identifier=response_identifier,
    )

    template = SemanticEvaluationCompletionTemplate.create(eval=eval, reasoning_effort=ReasoningEffort.MEDIUM)

    rating = "4"
    explanation = "The evaluation shows moderate quality."
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
                "query_identifier": eval.query_identifier,
                "context_identifier": eval.context_identifier,
                "response_identifier": eval.response_identifier,
                "query": query,
                "context": context,
                "reference_answer": reference_answer,
                "eval_criteria": eval.criteria,
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.75


@pytest.mark.asyncio
async def test_semantic_evaluation_completion_template_with_default_semantic_evals() -> None:
    """Test SemanticEvaluationCompletionTemplate with default semantic eval configurations."""
    from tlm.utils.scoring.semantic_evaluation_scoring_utils import DEFAULT_RAG_EVALS

    # Test context_sufficiency (has query and context identifiers)
    context_sufficiency_eval = DEFAULT_RAG_EVALS[0]
    template = SemanticEvaluationCompletionTemplate.create(
        eval=context_sufficiency_eval, reasoning_effort=ReasoningEffort.MEDIUM
    )

    rating = "5"
    explanation = "The document contains all necessary information."
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
                "query_identifier": context_sufficiency_eval.query_identifier,
                "context_identifier": context_sufficiency_eval.context_identifier,
                "response_identifier": context_sufficiency_eval.response_identifier,
                "query": "How much water does your Simple Water Bottle hold?",
                "context": "The Simple Water Bottle is a reusable 27 oz water bottle.",
                "reference_answer": None,
                "eval_criteria": context_sufficiency_eval.criteria,
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 1.0

    # Test response_groundedness (has query, context, and response identifiers)
    response_groundedness_eval = DEFAULT_RAG_EVALS[1]
    template = SemanticEvaluationCompletionTemplate.create(
        eval=response_groundedness_eval, reasoning_effort=ReasoningEffort.MEDIUM
    )

    rating = "3"
    explanation = "Some claims are supported, but not all."
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
                "query_identifier": response_groundedness_eval.query_identifier,
                "context_identifier": response_groundedness_eval.context_identifier,
                "response_identifier": response_groundedness_eval.response_identifier,
                "query": "How much water does your Simple Water Bottle hold?",
                "context": "The Simple Water Bottle is a reusable 27 oz water bottle.",
                "reference_answer": "The Simple Water Bottle holds 27 oz of water.",
                "eval_criteria": response_groundedness_eval.criteria,
                "max_explanation_words": 100,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) == explanation
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.5

    # Test query_ease (only has query identifier)
    query_ease_eval = DEFAULT_RAG_EVALS[3]
    template = SemanticEvaluationCompletionTemplate.create(eval=query_ease_eval, reasoning_effort=ReasoningEffort.NONE)

    rating = "4"
    llm_response = f"<rating>\n{rating}\n</rating>"

    with patch_acompletion(llm_response):
        completion = await generate_completion(
            template,
            template_kwargs={
                "query_identifier": query_ease_eval.query_identifier,
                "context_identifier": query_ease_eval.context_identifier,
                "response_identifier": query_ease_eval.response_identifier,
                "query": "How much water does your Simple Water Bottle hold?",
                "context": None,
                "reference_answer": None,
                "eval_criteria": query_ease_eval.criteria,
                "max_explanation_words": 0,
            },
        )

    assert isinstance(completion, Completion)
    assert completion.response_fields.get(ExtractedResponseField.SCORE) == rating
    assert completion.response_fields.get(ExtractedResponseField.EXPLANATION) is None
    assert completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE) == 0.75
