from difflib import SequenceMatcher
import re
import warnings

from tlm.types import Completion, ExtractedResponseField


def constrain_output(
    completion: Completion,
    original_message: str,
    constrain_outputs: list[str],
) -> None:
    """Match the original response to one of the constrained output options.
    Extract the provided output values using regex patterns.
    Returns last extracted value if multiple exist.
    If no value out of the possible `constrain_outputs` is directly mentioned in the response,
    the value with greatest string similarity to the response is returned (along with a warning).
    If there are no close matches between the LLM response and any of the possible `constrain_outputs`,
    then the last entry of the `constrain_outputs` list is returned.

    Params
    ------
    original_message: LLM response
    constrain_outputs: List of possible output options
    """
    escaped_constrain_outputs = [re.escape(output) for output in constrain_outputs]
    constrain_outputs_pattern = "(" + "|".join(escaped_constrain_outputs) + ")"

    # Parse category if LLM response is properly formatted
    exact_matches = re.findall(constrain_outputs_pattern, original_message, re.IGNORECASE)
    if len(exact_matches) > 0:
        # Find the corresponding original case-sensitive value from CONFIG.constrain_outputs
        matched_value = next(output for output in constrain_outputs if output.lower() == exact_matches[-1].lower())

        completion.add_response_field(ExtractedResponseField.ANSWER, matched_value)
        return

    # If there are no exact matches to a specific category, return the closest category based on string similarity.
    best_match = max(constrain_outputs, key=lambda x: SequenceMatcher(None, original_message, x).ratio())
    similarity_score = SequenceMatcher(None, original_message, best_match).ratio()

    # The 0.7 threshold is arbitrary and we can tune it later if needed based on the logs.
    if similarity_score < 0.7:
        best_match = constrain_outputs[-1]
        warning_message = (
            f"None of the constrain_outputs remotely match raw LLM output: {original_message}.\n"
            f"Best match was '{best_match}' with similarity score {similarity_score:.2f}.\n"
            f"Returning the last entry in the constrain outputs list: {best_match}."
        )
    else:
        warning_message = (
            f"None of the constrain_outputs exactly match raw LLM output: {original_message}.\n"
            f"Using closest match '{best_match}' with similarity score {similarity_score:.2f}"
        )

    completion.add_response_field(ExtractedResponseField.ANSWER, best_match)
    _scale_answer_confidence(completion, similarity_score)

    warnings.warn(warning_message)


def _scale_answer_confidence(completion: Completion, similarity_score: float) -> None:
    reduction_factor = 1e-3 if similarity_score < 0.7 else similarity_score

    if completion.perplexity is not None:
        completion.perplexity = completion.perplexity * reduction_factor
