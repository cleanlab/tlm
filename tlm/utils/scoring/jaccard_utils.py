import ast
import re
from typing import Any, List, Set


def extract_words(string: str) -> List[str]:
    """Extract words from string, with punctuation removed."""
    return re.findall(r"\b\w+\b", string)


def jaccard_similarity(
    answer: str,
    comparison: str,
    structured_outputs: bool = False,
) -> float:
    """Computes jaccard similarity between two strings.

    If structured_outputs is True, keys in the output JSON are ignored (as there will be a lot of overlap in the keys).
    The formula is: (I - S) / (U - S)
    I = length of the intersection
    U = length of the union
    S = length of the JSON keys / "structured" part
    """

    answer_words = set(extract_words(answer))
    comparison_words = set(extract_words(comparison))

    if structured_outputs:
        structure_keys = get_structured_output_keys(answer)

        return float(
            max(0, len(answer_words.intersection(comparison_words)) - len(structure_keys))
            / max(1, len(answer_words.union(comparison_words)) - len(structure_keys))
        )

    return float(len(answer_words.intersection(comparison_words)) / max(1, len(answer_words.union(comparison_words))))


def get_structured_output_keys(answer: str) -> Set[str]:
    try:
        answer_dict = ast.literal_eval(answer)
        return get_all_keys(answer_dict)
    except Exception:
        return set()


def get_all_keys(d: Any) -> Set[str]:
    keys = set()
    if isinstance(d, dict):
        for key, value in d.items():
            keys.add(key)
            keys.update(get_all_keys(value))
    elif isinstance(d, (list, tuple)):
        for item in d:
            keys.update(get_all_keys(item))
    return keys
