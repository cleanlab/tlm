import pytest

from tlm.utils.scoring.jaccard_utils import get_structured_output_keys, jaccard_similarity


@pytest.mark.parametrize(
    "answer, comparison, expected",
    [
        ("Hello, world!", "Hello, world!", 1.0),
        ("Hello, world!", "Hello, universe!", 1 / 3),
        ("Hello, world!", "Hello, world!?", 1.0),
        (
            "The quick brown fox jumps over the lazy dog.",
            "The swift black cat hops around the sleepy frog.",
            1 / 8,
        ),  # is there a reason why we don't ignore capitalization?
        ("Hello world", "Goodbye universe", 0.0),
    ],
)
def test_jaccard_similarity(answer: str, comparison: str, expected: float) -> None:
    assert jaccard_similarity(answer, comparison) == expected


@pytest.mark.parametrize(
    "answer, comparison, expected",
    [
        ("{'name': 'John', 'age': 30}", "{'name': 'John', 'age': 30}", 1.0),
        ("{'name': 'John', 'age': 30}", "{'name': 'Jane', 'age': 25}", 0.0),
        ("{'name': 'John', 'age': 30}", "{'name': 'John', 'age': 30, 'city': 'New York'}", 0.4),
        (
            "{'people': [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}], 'city': 'New York'}",
            "{'people': [{'name': 'Mary', 'age': 30}, {'name': 'Jane', 'age': 27}], 'city': 'Los Angeles'}",
            0.2,
        ),
    ],
)
def test_jaccard_similarity_structured_outputs(answer: str, comparison: str, expected: float) -> None:
    assert jaccard_similarity(answer, comparison, structured_outputs=True) == expected


def test_get_structured_output_keys_error() -> None:
    assert get_structured_output_keys("-1}") == set()
