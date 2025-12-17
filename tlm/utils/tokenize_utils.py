import functools

import numpy as np
import tiktoken

from tlm.config.defaults import get_settings
from tlm.config.models import DEFAULT_MODEL, ENCODING_MODELS
from tlm.config.presets import REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS, ReasoningEffort

settings = get_settings()


def get_max_words_for_observed_consistency_explanation(reasoning_effort: ReasoningEffort) -> int:
    """Explanation for observed consistency is limited to max_words to prevent token overflow which is calculated using length of reference answer.
    Params:
       reference_answers: One or more TLM answers to initial prompt. Reference answers could be multiple
       when quality of TLM allows for picking the most confident reference answer out of multiple.
    Returns:
        max_words_for_explanation: Max number of words encouraged in the Explanation of an Observed Consistency prompt s.t.
        the output token limit from the TLM is not hit. Value slightly underestimated for buffer using explanation_length_underestimate_factor.
    """
    num_words_for_answer = 50  # TODO: arbitrary number for now
    max_words_for_explanation = np.min(
        [
            (settings.MAX_TOKENS * settings.AVG_WORDS_PER_TOKEN) - num_words_for_answer,
            REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS[reasoning_effort],
        ]
    )
    allowed_words_for_explanation = int(
        np.ceil(
            max(
                10,
                max_words_for_explanation * settings.EXPLANATION_LENGTH_UNDERESTIMATE_FACTOR,
            )
        )
    )  # Minimally at least 10 words for explanation. TODO: consider capping the max-value of max_words as well once there are higher token limits.

    return round_max_words(allowed_words_for_explanation)


def round_max_words(max_words: int) -> int:
    """Round max words down to nearest 10 if < 100, otherwise to nearest 50."""
    if max_words < 10:
        return max_words
    elif max_words < 100:
        return (max_words // 10) * 10
    else:
        return (max_words // 50) * 50


@functools.lru_cache(maxsize=50)
def get_token_count(input: str, model: str) -> int:
    """Gets token count for given input."""
    enc = tiktoken.get_encoding(ENCODING_MODELS.get(model, ENCODING_MODELS[DEFAULT_MODEL]))
    return len(enc.encode(input, disallowed_special=()))
