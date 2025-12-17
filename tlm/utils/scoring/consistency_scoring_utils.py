import asyncio
import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from tlm.types import Completion, CompletionFailure
from tlm.templates.llm_consistency_completion_templates import (
    CodeConsistencyCompletionTemplate,
    StatementConsistencyCompletionTemplate,
)
from tlm.utils.errors import LLMConsistencyInferenceError
from tlm.utils.math_utils import compute_cosine_similarity, get_median_indices, get_nan_safe_mean
from tlm.utils.openai_utils import get_openai_client, get_text_embedding
from tlm.utils.scoring.jaccard_utils import jaccard_similarity
from tlm.utils.scoring.llm_consistency_scoring_utils import get_llm_consistency_scores
from tlm.utils.scoring.indicator_scoring_utils import compute_indicator_scores
from tlm.types import SimilarityMeasure

REFERENCE_COLUMN_NAME = "reference"
COMPARISON_COLUMN_NAME = "comparison"

LLM_CONSISTENCY_JACCARD_WEIGHT = 0.05

EMBEDDING_MODELS = {
    SimilarityMeasure.EMBEDDING_SMALL: "text-embedding-3-small",
    SimilarityMeasure.EMBEDDING_LARGE: "text-embedding-3-large",
}


async def compute_consistency_scores(
    reference_answers: list[str],
    comparison_answers: list[str],
    similarity_measure: SimilarityMeasure,
    structured_outputs: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generates consistency scores for QA tasks by computing similarity scores between reference and comparison answers.

    Returns array of average scores for each reference answer.
    """
    return await _compute_scores_qa(reference_answers, comparison_answers, similarity_measure, structured_outputs)


def compute_consistency_scores_classification(
    reference_answers: list[str],
    comparison_answers: list[str],
    comparison_completions: list[Completion | CompletionFailure],
    constrain_outputs: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generates scores for classification tasks by computing consistency scores.

    For classification tasks:
    - If model returns logprobs, compute consistency scores using _generate_consistency_scores_classification
    - If model does not return logprobs, compute consistency scores using _generate_indicator_scores
    - Indicator scores are always None for classification tasks

    Returns list of consistency scores, one for each reference answer, and flattened list of all consistency scores in row-major order by reference answer.
    """
    logprobs_available = any(
        comparison_completion.perplexity is not None
        for comparison_completion in comparison_completions
        if isinstance(comparison_completion, Completion)
    )

    if logprobs_available:
        average_consistency_scores, consistency_scores_flat = _generate_consistency_scores_classification(
            reference_answers, comparison_answers, comparison_completions, constrain_outputs
        )
    else:
        # For models without logprobs, use indicator scores as consistency scores
        average_consistency_scores, consistency_scores_flat = compute_indicator_scores(
            reference_answers, comparison_answers
        )

    return average_consistency_scores, consistency_scores_flat


def _generate_consistency_scores_classification(
    reference_answers: Sequence[str],
    comparison_answers: list[str],
    comparison_completions: Sequence[Completion | CompletionFailure],
    constrain_outputs: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generates consistency scores for classification tasks by comparing reference answers with comparison answers.

    For each reference answer and comparison answer pair:
    - If comparison answer matches reference answer exactly, score = perplexity of comparison answer
    - If comparison answer does not match or cannot be parsed, score = 0

    Returns array of average scores for each reference answer.
    """
    scores = []

    for i, ref_answer in enumerate(reference_answers):
        # If reference answer didn't match constrain outputs, score is 0
        if ref_answer not in constrain_outputs:
            scores.append([0.0] * len(comparison_completions))
            continue

        observed_consistency_scores = []
        for idx, completion in enumerate(comparison_completions):
            # Handle completion failures and parsing failures by assigning 0
            if isinstance(completion, CompletionFailure):
                observed_consistency_scores.append(0.0)
                continue

            # For successful completions, check for exact match
            comp_answer = comparison_answers[idx]
            if comp_answer == ref_answer and (score := completion.perplexity) is not None:
                observed_consistency_scores.append(score)
            else:
                if len(constrain_outputs) == 2:
                    if comp_answer in constrain_outputs and (raw_score := completion.perplexity) is not None:
                        score = 1 - raw_score
                    else:
                        # If the completion doesn't match the constrain outputs, we abstain from scoring by giving a neutral score of 0.5
                        score = 0.5
                else:
                    score = 0.0

                observed_consistency_scores.append(score)

        scores.append(observed_consistency_scores)

    scores_array = np.array(scores)
    return get_nan_safe_mean(scores_array, axis=1, expected_array_length=len(reference_answers)), scores_array.flatten()


async def _compute_scores_qa(
    reference_answers: list[str],
    comparison_answers: list[str],
    similarity_measure: SimilarityMeasure,
    structured_outputs: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if similarity_measure == SimilarityMeasure.JACCARD:
        scores = _compute_jaccard_similarity_scores(reference_answers, comparison_answers, structured_outputs)
    elif similarity_measure in [SimilarityMeasure.EMBEDDING_SMALL, SimilarityMeasure.EMBEDDING_LARGE]:
        embedding_model = EMBEDDING_MODELS[similarity_measure]
        scores = await _compute_embedding_similarity_scores(reference_answers, comparison_answers, embedding_model)
    elif similarity_measure == SimilarityMeasure.CODE:
        scores = await _compute_code_similarity_scores(reference_answers, comparison_answers)
    elif similarity_measure == SimilarityMeasure.STATEMENT:
        scores = await _compute_statement_similarity_scores(reference_answers, comparison_answers)

    # compute mean consistency score for each reference answer, return
    return get_nan_safe_mean(
        scores.reshape((len(reference_answers), -1)),
        axis=1,
        expected_array_length=len(reference_answers),
    ), scores


def _get_comparison_pairs(
    reference_answers: Sequence[str | list[float]], comparison_answers: Sequence[str | list[float]]
) -> pd.DataFrame:
    return pd.DataFrame(
        [*itertools.product(reference_answers, comparison_answers)],
        columns=[REFERENCE_COLUMN_NAME, COMPARISON_COLUMN_NAME],
    )


def _compute_jaccard_similarity_scores(
    reference_answers: list[str], comparison_answers: list[str], structured_outputs: bool = False
) -> npt.NDArray[np.float64]:
    comparison_pairs = _get_comparison_pairs(reference_answers, comparison_answers)
    return np.array(
        [
            jaccard_similarity(reference, comparison, structured_outputs)
            for reference, comparison in comparison_pairs.itertuples(index=False)
        ]
    )


async def _compute_embedding_similarity_scores(
    reference_answers: list[str], comparison_answers: list[str], embedding_model: str
) -> npt.NDArray[np.float64]:
    async with get_openai_client() as openai_client:
        reference_embedding_tasks = [
            asyncio.create_task(get_text_embedding(openai_client, reference, embedding_model))
            for reference in reference_answers
        ]

        comparison_embedding_tasks = [
            asyncio.create_task(get_text_embedding(openai_client, comparison, embedding_model))
            for comparison in comparison_answers
        ]

        reference_embeddings, comparison_embeddings = await asyncio.gather(
            asyncio.gather(*reference_embedding_tasks),
            asyncio.gather(*comparison_embedding_tasks),
        )

        comparison_pairs = _get_comparison_pairs(reference_embeddings, comparison_embeddings)

    return np.array(
        [
            compute_cosine_similarity(reference, comparison)
            for reference, comparison in comparison_pairs.itertuples(index=False)
        ]
    )


async def _compute_code_similarity_scores(
    reference_answers: list[str],
    comparison_answers: list[str],
) -> npt.NDArray[np.float64]:
    jaccard_scores = _compute_jaccard_similarity_scores(reference_answers, comparison_answers)
    jaccard_full_matrix = jaccard_scores.reshape(len(reference_answers), -1)

    try:
        median_indices = get_median_indices(jaccard_full_matrix)
        median_jaccard_scores = jaccard_full_matrix[np.arange(len(reference_answers)), median_indices]

        llm_consistency_scores = np.full(len(reference_answers), np.nan)
        is_identical_mask = median_jaccard_scores == 1.0
        llm_consistency_scores[is_identical_mask] = 1.0

        if (~is_identical_mask).any():
            comparison_answers_subset = [comparison_answers[i] for i in median_indices]

            non_identical_llm_consistency_scores: npt.NDArray[np.float64] = await get_llm_consistency_scores(
                np.array(reference_answers)[~is_identical_mask].tolist(),
                np.array(comparison_answers_subset)[~is_identical_mask].tolist(),
                CodeConsistencyCompletionTemplate.create(),
            )
            llm_consistency_scores[~is_identical_mask] = non_identical_llm_consistency_scores

        llm_consistency_matrix = np.tile(llm_consistency_scores.reshape(-1, 1), jaccard_full_matrix.shape[1])

        weighted_scores = np.where(
            np.isnan(llm_consistency_matrix),
            jaccard_full_matrix,
            jaccard_full_matrix * LLM_CONSISTENCY_JACCARD_WEIGHT
            + llm_consistency_matrix * (1 - LLM_CONSISTENCY_JACCARD_WEIGHT),
        )

        return weighted_scores.flatten()

    except LLMConsistencyInferenceError as e:
        print(f"LLM consistency failed, falling back to jaccard similarity: {e}")
        return jaccard_full_matrix.flatten()


async def _compute_statement_similarity_scores(
    reference_answers: list[str],
    comparison_answers: list[str],
) -> npt.NDArray[np.float64]:
    jaccard_scores = _compute_jaccard_similarity_scores(reference_answers, comparison_answers)
    discrepancy_scores = await get_llm_consistency_scores(
        reference_answers,
        comparison_answers,
        StatementConsistencyCompletionTemplate.create(),
    )

    try:
        weighted_scores = np.where(
            np.isnan(discrepancy_scores),
            jaccard_scores,
            jaccard_scores * LLM_CONSISTENCY_JACCARD_WEIGHT + discrepancy_scores * (1 - LLM_CONSISTENCY_JACCARD_WEIGHT),
        )
        return weighted_scores
    except LLMConsistencyInferenceError as e:
        print(f"LLM consistency failed, falling back to jaccard similarity: {e}")
        return jaccard_scores
