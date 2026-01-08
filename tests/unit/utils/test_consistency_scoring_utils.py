from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.helpers.embedding_fixtures import fake_embedding
from tlm.utils.scoring.consistency_scoring_utils import EMBEDDING_MODELS, compute_consistency_scores
from tlm.types import SimilarityMeasure


@pytest.mark.asyncio
async def test_compute_scores_qa_jaccard() -> None:
    reference_answers = ["Hello, world!", "Hello", "Hello, world!"]
    comparison_answers = ["Hello, world", "Hello, universe", "Hello, universe!"]
    similarity_measure = SimilarityMeasure.JACCARD
    avg_scores, scores = await compute_consistency_scores(reference_answers, comparison_answers, similarity_measure)
    assert np.allclose(scores, np.array([1, 1 / 3, 1 / 3, 0.5, 0.5, 0.5, 1, 1 / 3, 1 / 3]))
    assert np.allclose(avg_scores, np.array([5 / 9, 0.5, 5 / 9]))


@pytest.mark.parametrize("similarity_measure", [SimilarityMeasure.EMBEDDING_SMALL, SimilarityMeasure.EMBEDDING_LARGE])
@pytest.mark.asyncio
async def test_compute_scores_qa_embedding(similarity_measure: SimilarityMeasure) -> None:
    reference_answers = ["Hello, world!", "Hello", "Hello, world!"]
    comparison_answers = ["Hello, world!", "Hello, universe", "Hello, universe!"]

    # Create a mock OpenAI client with embeddings.create method
    mock_openai_client = MagicMock()
    embedding_calls: list[tuple[str, str]] = []

    async def mock_embeddings_create(input: str, model: str, timeout: float) -> MagicMock:
        embedding_calls.append((input, model))
        response = MagicMock()
        response.data = [MagicMock(embedding=fake_embedding(input, 3))]
        return response

    mock_openai_client.embeddings.create = mock_embeddings_create

    @asynccontextmanager
    async def mock_get_openai_client():
        yield mock_openai_client

    with patch(
        "tlm.utils.scoring.consistency_scoring_utils.get_openai_client",
        mock_get_openai_client,
    ):
        avg_scores, scores = await compute_consistency_scores(reference_answers, comparison_answers, similarity_measure)
        assert scores[0] == 1
        assert scores[6] == 1
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
        assert np.all(avg_scores >= 0)
        assert np.all(avg_scores <= 1)
        assert len(embedding_calls) == len(reference_answers) + len(comparison_answers)
        for text, model in embedding_calls:
            assert text in reference_answers or text in comparison_answers
            assert model == EMBEDDING_MODELS[similarity_measure]
