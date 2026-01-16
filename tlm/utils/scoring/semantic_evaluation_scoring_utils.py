import numpy as np

from tlm.utils.math_utils import get_nan_safe_mean, make_score_asymptotic
from tlm.types import Completion, CompletionFailure, ExtractedResponseField, Eval
from tlm.utils.parse_utils import compute_score_expected_value


_DEFAULT_EVALS_DICT: list[dict[str, str | None]] = [
    {
        "name": "context_sufficiency",
        "criteria": "Determine if the Document contains 100% of the information needed to answer the Question. If any external knowledge or assumptions are required, it does not meet the criteria. Each Question component must have explicit support in the Document.",
        "query_identifier": "Question",
        "context_identifier": "Document",
        "response_identifier": None,
    },
    {
        "name": "response_groundedness",
        "criteria": "Review the Response to the Query and assess whether every factual claim in the Response is explicitly supported by the provided Context. A Response meets the criteria if all information is directly backed by evidence in the Context, without relying on assumptions, external knowledge, or unstated inferences. The focus is on whether the Response is fully grounded in the Context, rather than whether it fully addresses the Query. If any claim in the Response lacks direct support or introduces information not present in the Context, the Response is bad and does not meet the criteria.",
        "query_identifier": "Query",
        "context_identifier": "Context",
        "response_identifier": "Response",
    },
    {
        "name": "response_helpfulness",
        "criteria": """Assess whether the AI Assistant Response is a helpful answer to the User Query.
A Response is considered helpful if it makes a genuine attempt to answer the question, even if the answer is incorrect or incomplete. Factual inaccuracies should not affect the assessment. The only thing that matters is whether the Assistant tries to answer the question.
A Response is considered not helpful if it avoids answering the question. For example, by saying or implying things like "I don't know", "Sorry", "No information available", or any other form of refusal or deflection.""",
        "query_identifier": "User Query",
        "context_identifier": None,
        "response_identifier": "AI Assistant Response",
    },
    {
        "name": "query_ease",
        "criteria": """Determine whether the above User Request appears simple and straightforward.
A bad User Request will appear either: ambiguous in intent, complex, purposefully tricky, abnormal, or disgruntled.
A good User Request is phrased clearly and expresses an achievable intent. Basic conversational and non-propositional statements are also considered good.
Should an AI Assistant be able to properly answer the User Request, it is considered good. The AI Assistant handling this User Request has additional knowledge about: the user, domain-specific terms and abbreviations, and any necessary factual information. So a User Request missing information could still be good; vagueness due to undefined pronouns/terms or references to unknown context does not make a User Request bad.
""",
        "query_identifier": "User Request",
        "context_identifier": None,
        "response_identifier": None,
    },
]

DEFAULT_RAG_EVALS = [Eval(**eval_dict) for eval_dict in _DEFAULT_EVALS_DICT]  # type: ignore


def compute_semantic_evaluation_scores(
    reference_answers: list[str | None],
    evals: list[Eval],
    semantic_evaluation_completions: list[Completion | CompletionFailure],
) -> dict[str, float]:
    """
    Computes scores for semantic evaluations across all reference answers based on the chat completions.
    If reference answers are not provided, the evaluations did not require the response.
    Returns a list of scores, one for each eval, in the same order as the input.
    """
    raw_scores = np.vectorize(_get_score_from_semantic_eval_completion)(semantic_evaluation_completions)

    if not reference_answers or len(reference_answers) == 1:
        # can map directly
        scores_array = raw_scores
    else:
        raw_scores = raw_scores.reshape(-1, len(reference_answers), order="F")
        scores_array = get_nan_safe_mean(raw_scores, axis=1, expected_array_length=len(reference_answers))

    scores_array = make_score_asymptotic(scores_array)
    scores_with_none = np.where(np.isnan(scores_array), None, scores_array)  # type: ignore
    eval_scores = {eval.name: score for eval, score in zip(evals, scores_with_none)}

    return eval_scores


def _get_score_from_semantic_eval_completion(
    completion: Completion | CompletionFailure,
) -> float:
    """Generate semantic eval score for a given reference answer and eval."""
    if isinstance(completion, CompletionFailure):
        return np.nan

    if raw_score := completion.response_fields.get(ExtractedResponseField.SCORE):
        if (weighted_score := compute_score_expected_value(completion, raw_score)) is not None:
            return weighted_score

    if mapped_score := completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE):
        return mapped_score

    return np.nan
