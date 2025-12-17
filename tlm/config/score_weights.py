from typing import Dict

from tlm.config.models import (
    CLAUDE_3_5_SONNET,
    CLAUDE_3_HAIKU,
    CLAUDE_3_SONNET,
    DEFAULT_MODEL,
    GPT_3_5_TURBO_16K,
    GPT_4,
    GPT_4O,
    GPT_4O_MINI,
    O1_PREVIEW,
)
from tlm.config.presets import WorkflowType

CONSISTENCY_SCORE_WEIGHT = "consistency_score_weight"
INDICATOR_SCORE_WEIGHT = "indicator_score_weight"
SELF_REFLECTION_SCORE_WEIGHT = "self_reflection_score_weight"
PERPLEXITY_SCORE_WEIGHT = "perplexity_score_weight"
PROMPT_EVAL_SCORE_WEIGHT = "prompt_eval_score_weight"

# score weighting for various models
# for calls where perplexity is not used, that value will be set to None
COMPONENT_SCORE_WEIGHTS: Dict[WorkflowType, Dict[str, Dict[str, float]]] = {
    WorkflowType.DEFAULT: {
        # No prompt eval for default workflow yet so set the weight to 0
        DEFAULT_MODEL: {
            CONSISTENCY_SCORE_WEIGHT: 0.32,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.47,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0.2,
        },
        GPT_3_5_TURBO_16K: {
            CONSISTENCY_SCORE_WEIGHT: 0.51,
            INDICATOR_SCORE_WEIGHT: 0.05,
            SELF_REFLECTION_SCORE_WEIGHT: 0.17,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0.27,
        },
        GPT_4: {
            CONSISTENCY_SCORE_WEIGHT: 0.48,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.3,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0.21,
        },
        GPT_4O: {
            CONSISTENCY_SCORE_WEIGHT: 0.68,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.2,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0.11,
        },
        GPT_4O_MINI: {
            CONSISTENCY_SCORE_WEIGHT: 0.51,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.3,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0.18,
        },
        O1_PREVIEW: {
            CONSISTENCY_SCORE_WEIGHT: 0.69,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.3,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0,  # o1 do not return logprobs yet
        },
        CLAUDE_3_HAIKU: {
            CONSISTENCY_SCORE_WEIGHT: 0.69,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.3,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0,  # claude models do not return logprobs
        },
        CLAUDE_3_SONNET: {
            CONSISTENCY_SCORE_WEIGHT: 0.59,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.4,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0,  # claude models do not return logprobs
        },
        CLAUDE_3_5_SONNET: {
            CONSISTENCY_SCORE_WEIGHT: 0.4,
            INDICATOR_SCORE_WEIGHT: 0.01,
            SELF_REFLECTION_SCORE_WEIGHT: 0.59,
            PROMPT_EVAL_SCORE_WEIGHT: 0.0,
            PERPLEXITY_SCORE_WEIGHT: 0,  # claude models do not return logprobs
        },
    },
    WorkflowType.RAG: {
        DEFAULT_MODEL: {
            CONSISTENCY_SCORE_WEIGHT: 0.1,
            INDICATOR_SCORE_WEIGHT: 0.001,
            SELF_REFLECTION_SCORE_WEIGHT: 0.639,
            PROMPT_EVAL_SCORE_WEIGHT: 0.06,
            PERPLEXITY_SCORE_WEIGHT: 0.2,
        },
        O1_PREVIEW: {
            CONSISTENCY_SCORE_WEIGHT: 0.353,
            INDICATOR_SCORE_WEIGHT: 0.009,
            SELF_REFLECTION_SCORE_WEIGHT: 0.578,
            PROMPT_EVAL_SCORE_WEIGHT: 0.06,
            PERPLEXITY_SCORE_WEIGHT: 0,
        },
        CLAUDE_3_HAIKU: {
            CONSISTENCY_SCORE_WEIGHT: 0.353,
            INDICATOR_SCORE_WEIGHT: 0.009,
            SELF_REFLECTION_SCORE_WEIGHT: 0.578,
            PROMPT_EVAL_SCORE_WEIGHT: 0.06,
            PERPLEXITY_SCORE_WEIGHT: 0,
        },
        CLAUDE_3_SONNET: {
            CONSISTENCY_SCORE_WEIGHT: 0.353,
            INDICATOR_SCORE_WEIGHT: 0.009,
            SELF_REFLECTION_SCORE_WEIGHT: 0.578,
            PROMPT_EVAL_SCORE_WEIGHT: 0.06,
            PERPLEXITY_SCORE_WEIGHT: 0,
        },
        CLAUDE_3_5_SONNET: {
            CONSISTENCY_SCORE_WEIGHT: 0.353,
            INDICATOR_SCORE_WEIGHT: 0.009,
            SELF_REFLECTION_SCORE_WEIGHT: 0.578,
            PROMPT_EVAL_SCORE_WEIGHT: 0.06,
            PERPLEXITY_SCORE_WEIGHT: 0,
        },
    },
}
