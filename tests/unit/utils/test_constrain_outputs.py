from tlm.types import Completion, ExtractedResponseField
from tlm.utils.constrain_outputs_utils import constrain_output


def test_constrain_output_exact_match() -> None:
    completion = Completion.from_response({"response": "The capital of France is Paris."})
    constrain_output(completion, "The capital of France is Paris.", ["Paris", "London"])
    assert completion.response_fields[ExtractedResponseField.ANSWER] == "Paris"


def test_constrain_output_match_case_insensitive() -> None:
    completion = Completion.from_response({"response": "The capital of France is Paris."})
    constrain_output(completion, "The capital of France is paris.", ["paris", "tokyo"])
    assert completion.response_fields[ExtractedResponseField.ANSWER] == "paris"


def test_constrain_output_fallback_match() -> None:
    completion = Completion.from_response({"response": "The capital of France is Paris."})
    constrain_output(completion, "The capital of France is Paris.", ["Berlin", "London"])
    assert completion.response_fields[ExtractedResponseField.ANSWER] == "London"


def test_constrain_output_closest_match() -> None:
    completion = Completion.from_response({"response": "colour"})
    constrain_output(completion, "colour", ["weight", "color", "style"])
    assert completion.response_fields[ExtractedResponseField.ANSWER] == "color"
