from enum import Enum
from typing import Any


class QualityPreset(str, Enum):
    """Quality presets that control the trade-off between speed and accuracy.

    Higher quality presets generate more completions and use more advanced techniques,
    resulting in higher trustworthiness scores but slower inference and higher costs.

    Values:
        `BASE`, `LOW`, `MEDIUM` (default), `HIGH`, `BEST`
    """

    BASE = "base"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BEST = "best"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels that control explanation generation for trustworthiness scores.

    Higher reasoning effort generates longer explanations that provide more detailed
    reasoning about why a particular trustworthiness score was assigned.

    Values:
        `NONE` (default), `LOW`, `MEDIUM`, `HIGH`
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS = {
    ReasoningEffort.NONE: 0,
    ReasoningEffort.LOW: int(40 / 0.8),
    ReasoningEffort.MEDIUM: int(100 / 0.8),
    ReasoningEffort.HIGH: int(300 / 0.8),
}


class WorkflowType(str, Enum):
    """Enum for different types of workflows supported by TLM."""

    QA = "qa"
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    RAG = "rag"
    STRUCTURED_OUTPUT_SCORING = "structured_output_scoring"

    DEFAULT = QA

    @classmethod
    def from_inference_params(
        cls,
        *,
        openai_args: dict[str, Any],
        rag: bool,
        score: bool,
        constrain_outputs: list[str] | None = None,
    ) -> "WorkflowType":
        if openai_args.get("response_format") is not None and score:
            return cls.STRUCTURED_OUTPUT_SCORING

        if rag:
            return cls.RAG

        if constrain_outputs:
            if len(constrain_outputs) == 2:
                return cls.BINARY_CLASSIFICATION
            return cls.CLASSIFICATION

        return cls.QA


ALTERNATE_REFERENCE_TEMPERATURE = "alternate_reference_temperature"
MIN_REFERENCE_COMPLETIONS = "min_reference_completions"
MIN_CONSISTENCY_COMPLETIONS = "min_consistency_completions"
MIN_SELF_REFLECTION_COMPLETIONS = "min_self_reflection_completions"
REASONING_EFFORT = "reasoning_effort"
USE_PROMPT_EVALUATION = "use_prompt_evaluation"

DEFAULT_CONFIG_FOR_QUALITY: dict[QualityPreset, dict[str, Any]] = {
    QualityPreset.BEST: {
        ALTERNATE_REFERENCE_TEMPERATURE: 1.0,
        MIN_REFERENCE_COMPLETIONS: 2,
        MIN_CONSISTENCY_COMPLETIONS: 1,
        MIN_SELF_REFLECTION_COMPLETIONS: 1,
    },
    QualityPreset.HIGH: {
        ALTERNATE_REFERENCE_TEMPERATURE: 1.0,
        MIN_REFERENCE_COMPLETIONS: 2,
        MIN_CONSISTENCY_COMPLETIONS: 1,
        MIN_SELF_REFLECTION_COMPLETIONS: 1,
    },
    QualityPreset.MEDIUM: {
        ALTERNATE_REFERENCE_TEMPERATURE: 0.0,
        MIN_REFERENCE_COMPLETIONS: 1,
        MIN_CONSISTENCY_COMPLETIONS: 1,
        MIN_SELF_REFLECTION_COMPLETIONS: 1,
    },
    QualityPreset.LOW: {
        ALTERNATE_REFERENCE_TEMPERATURE: 0.0,
        MIN_REFERENCE_COMPLETIONS: 1,
        MIN_CONSISTENCY_COMPLETIONS: 1,
        MIN_SELF_REFLECTION_COMPLETIONS: 1,
        REASONING_EFFORT: ReasoningEffort.NONE,
    },
    QualityPreset.BASE: {
        ALTERNATE_REFERENCE_TEMPERATURE: 0.0,
        MIN_REFERENCE_COMPLETIONS: 1,
        MIN_CONSISTENCY_COMPLETIONS: 0,
        MIN_SELF_REFLECTION_COMPLETIONS: 0,
        REASONING_EFFORT: ReasoningEffort.NONE,
        USE_PROMPT_EVALUATION: False,
    },
}

NUM_CONSISTENCY_COMPLETIONS = "num_consistency_completions"
NUM_SELF_REFLECTION_COMPLETIONS = "num_self_reflection_completions"

DEFAULT_CONFIG_FOR_QUALITY_AND_WORKFLOW = {
    QualityPreset.BEST: {WorkflowType.DEFAULT: {NUM_CONSISTENCY_COMPLETIONS: 8, NUM_SELF_REFLECTION_COMPLETIONS: -1}},
    QualityPreset.HIGH: {WorkflowType.DEFAULT: {NUM_CONSISTENCY_COMPLETIONS: 4, NUM_SELF_REFLECTION_COMPLETIONS: -1}},
    QualityPreset.MEDIUM: {WorkflowType.DEFAULT: {NUM_CONSISTENCY_COMPLETIONS: 0, NUM_SELF_REFLECTION_COMPLETIONS: -1}},
    QualityPreset.LOW: {
        WorkflowType.DEFAULT: {NUM_CONSISTENCY_COMPLETIONS: 0, NUM_SELF_REFLECTION_COMPLETIONS: -1},
        WorkflowType.RAG: {NUM_CONSISTENCY_COMPLETIONS: 0, NUM_SELF_REFLECTION_COMPLETIONS: 2},
    },
    QualityPreset.BASE: {
        WorkflowType.DEFAULT: {NUM_CONSISTENCY_COMPLETIONS: 0, NUM_SELF_REFLECTION_COMPLETIONS: 0},
    },
}
