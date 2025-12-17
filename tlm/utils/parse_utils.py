import logging
import math
import string
from typing import Dict, List

import numpy as np
from litellm.types.utils import ChatCompletionTokenLogprob

from tlm.utils.math_utils import _logprob_to_probability
from tlm.types import CompletionFailure, Completion

logger = logging.getLogger(__name__)


def compute_score_expected_value(completion: Completion | CompletionFailure, raw_score: str) -> float | None:
    """
    Calculate the expected value of the score.

    The formula used is:
    (1/4) * sum(Pr(i) * (i-1) for i in range(1, 6))

    Where:
    - Pr(i) is the probability of outcome i
    - i represents the score or value associated with outcome i

    This gives the average expected score across 5 possible outcomes,
    with each outcome weighted by its probability and shifted by -1.
    """
    if isinstance(completion, CompletionFailure):
        return np.nan

    try:
        completion_logprobs = completion.logprobs
        assert completion_logprobs is not None
        assert completion_logprobs.content is not None

        generated_logprobs = completion_logprobs.content
        # First, locate the token that we care for logprobs, i.e., the token matching the score
        score_logprobs = None
        for token_logprobs in generated_logprobs:
            if token_logprobs.token == raw_score:
                score_logprobs = token_logprobs
                break

        if score_logprobs is None:
            return None

        # Then, calculate the score based on the logprobs
        token_linear_probability: Dict[int, float] = {1: 1e-3, 2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 1e-3}

        for token_logprob in score_logprobs.top_logprobs:
            logprob = token_logprob.logprob

            # Filter out non-decimal token or tokens outside the 1-5 range
            if not token_logprob.token.isdecimal() or not (1 <= int(token_logprob.token) <= 5):
                continue

            # Calculate the linear probability
            linear_prob = math.exp(logprob)
            token_score = int(token_logprob.token)
            token_linear_probability[token_score] += linear_prob

        sum_of_weighted_scores = 0.0
        for score, prob in token_linear_probability.items():
            sum_of_weighted_scores += (score - 1) * prob

        # Scale the sum of linear probability to 1
        sum_linear_probability = sum(token_linear_probability.values())
        weighted_summed_score = (sum_of_weighted_scores / sum_linear_probability) / 4.0
        return weighted_summed_score
    except Exception as e:
        logger.exception("Failed to calculate weighted_summed_score for completion: %s: %s", completion, e)
        return None


def get_choice_token_confidence(completion: Completion) -> float | None:
    """Returns the confidence (probability of the top logprob) of the answer token.
    This function specifically extracts the probability of 'A'/'B' or True/False tokens for self-reflection scores.
    """
    if completion.template is None or completion.template.answer_choice_tokens is None:
        return None

    completion_logprobs = completion.logprobs

    if completion_logprobs is None or completion_logprobs.content is None:
        return None

    # define translator to remove punctuation from string
    translator = str.maketrans("", "", string.punctuation)

    # search in reverse order as the answer is most likely to be at the end
    # TODO: implement better logic instead of hardcoding all the tokens
    for token_logprob in reversed(completion_logprobs.content):
        preprocessed_token = token_logprob.token.strip().lower().translate(translator)
        for answer_choice_token in completion.template.answer_choice_tokens:
            if preprocessed_token == answer_choice_token.token.lower():
                token_confidence = _logprob_to_probability(token_logprob.logprob)
                if answer_choice_token.positive:
                    return token_confidence
                else:
                    return 1 - token_confidence

    return 0.5  # default score if token is not present in logprobs


def get_parsed_answer_tokens_confidence(
    completion: Completion,
    parsed_answer_start_idx: int,
    parsed_answer_end_idx: int,
) -> float | None:
    completion_logprobs = completion.logprobs
    assert completion_logprobs is not None
    assert completion_logprobs.content is not None

    per_position_completion_list = [
        completion for completion in completion_logprobs.content
    ]  # get top logprobs for each token
    if all(
        completion.top_logprobs for completion in per_position_completion_list
    ):  # if all top logprobs are returned correctly
        answer_token_confidence = _get_probability_of_generic_answer_tokens(
            completion.message,
            per_position_completion_list,
            parsed_answer_start_idx,
            parsed_answer_end_idx,
        )
        return answer_token_confidence
    return None


def _get_probability_of_generic_answer_tokens(
    message: str,
    per_position_completion_list: List[ChatCompletionTokenLogprob],
    parsed_answer_start_idx: int,
    parsed_answer_end_idx: int,
) -> float | None:
    if not message:  # if message is empty, then logprobs are not present
        return None
    tokens = [completion.token for completion in per_position_completion_list]
    top_logprobs_at_start_token_index = _find_token_index(tokens, parsed_answer_start_idx)
    top_logprobs_at_end_token_index = _find_token_index(tokens, parsed_answer_end_idx - 1)

    # Calculate mean probability across all tokens from start to end index
    probabilities = []
    for i in range(top_logprobs_at_start_token_index, top_logprobs_at_end_token_index + 1):
        probability = _logprob_to_probability(per_position_completion_list[i].logprob)
        normalized_token = _get_normalized_token(per_position_completion_list[i].token)
        if per_position_completion_list[i].top_logprobs:
            for top_logprob in per_position_completion_list[i].top_logprobs:
                if (
                    _get_normalized_token(top_logprob.token) == normalized_token
                    and top_logprob.token != per_position_completion_list[i].token
                ):
                    # also sum the logprobs if normalized tokens are the same
                    # for example, these would all be the same: "yes.", "Yes.", "Yes!"
                    probability += _logprob_to_probability(top_logprob.logprob)
        probability = min(probability, 1.0)
        probabilities.append(probability)
    if probabilities:
        return sum(probabilities) / len(probabilities)

    return 0.5


def _get_normalized_token(token_str: str) -> str:
    start = 0
    for start in range(len(token_str)):
        if not token_str[start].isspace() and token_str[start] not in string.punctuation:
            break
    end = len(token_str) - 1
    for end in range(len(token_str) - 1, -1, -1):
        if not token_str[end].isspace() and token_str[end] not in string.punctuation:
            break
    return token_str[start : end + 1].lower()


def _find_token_index(tokens: List[str], string_index: int) -> int:
    """Takes in a tokenized string and returns the index of the token that occurs at string_index of string created from joining the tokenized string together. Returns last index if string_index is out of bounds."""
    index_count = 0
    for idx, token in enumerate(tokens):
        token_length = len(token)
        if index_count + token_length > string_index:
            return idx
        index_count += token_length
    return -1


def compute_mean_message_confidence(completion: Completion) -> float:
    """Returns the mean confidence (probability) of all logprobs in the response message."""
    completion_logprobs = completion.logprobs
    assert completion_logprobs is not None
    assert completion_logprobs.content is not None

    logprobs = [completion.logprob for completion in completion_logprobs.content]
    mean_message_confidence = np.mean(_logprob_to_probability(logprobs))

    return float(mean_message_confidence)
