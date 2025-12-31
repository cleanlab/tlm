from pydantic import BaseModel, Field

from tlm.config.presets import (
    DEFAULT_CONFIG_FOR_QUALITY,
    DEFAULT_CONFIG_FOR_QUALITY_AND_WORKFLOW,
    QualityPreset,
    ReasoningEffort,
    WorkflowType,
)
from tlm.config.provider import ModelProvider
from tlm.types import SimilarityMeasure

from tlm.config.defaults import get_settings

settings = get_settings()


class ReferenceCompletionConfigInput(BaseModel):
    num_reference_completions: int | None = Field(
        default=None, description="The attempted number of reference completions to generate."
    )


class ReferenceCompletionConfig(BaseModel):
    num_reference_completions: int = 1
    min_reference_completions: int = Field(
        default=1, description="The minimum number of successful reference completions required."
    )
    alternate_reference_temperature: float | None = Field(
        description="Temperature used for reference completions after the first one."
    )


class ObservedConsistencyConfigInput(BaseModel):
    num_consistency_completions: int | None = Field(
        default=None, description="The attempted number of observed consistency completions to generate."
    )
    observed_consistency_temperature: float | None = None


class ObservedConsistencyConfig(BaseModel):
    num_consistency_completions: int
    min_consistency_completions: int = Field(
        description="The minimum number of successful observed consistency completions required."
    )
    observed_consistency_temperature: float = 1.0


class SelfReflectionConfigInput(BaseModel):
    self_reflection_temperature: float | None = None
    num_self_reflection_completions: int | None = Field(
        default=None,
        description=(
            "The number of self reflection prompts to use. Note that the first X number of prompts will be used, "
            "i.e. the order of the prompt templates in SELF_REFLECTION_TEMPLATES_BY_WORKFLOW[workflow_type] matters. "
            "-1 means all prompts will be used."
        ),
    )


class SelfReflectionConfig(BaseModel):
    self_reflection_temperature: float | None = None
    num_self_reflection_completions: int
    min_self_reflection_completions: int = Field(
        description="The minimum number of successful self reflection completions required."
    )


class SemanticEvalsConfigInput(BaseModel):
    use_prompt_evaluation: bool | None = None
    prompt_evaluation_temperature: float | None = None
    semantic_evaluation_temperature: float | None = None


class SemanticEvalsConfig(BaseModel):
    use_prompt_evaluation: bool = False
    prompt_evaluation_temperature: float = 0.0
    semantic_evaluation_temperature: float = 0.0


class ModelProviderInput(BaseModel):
    provider: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    api_version: str | None = None


class ConfigInput(
    ReferenceCompletionConfigInput,
    ObservedConsistencyConfigInput,
    SelfReflectionConfigInput,
    SemanticEvalsConfigInput,
    ModelProviderInput,
):
    quality_preset: QualityPreset = QualityPreset.MEDIUM
    reasoning_effort: ReasoningEffort | None = None
    similarity_measure: SimilarityMeasure | None = None
    constrain_outputs: list[str] | None = None


class BaseConfig(
    ReferenceCompletionConfig,
    ObservedConsistencyConfig,
    SelfReflectionConfig,
    SemanticEvalsConfig,
    ModelProvider,
):
    workflow_type: WorkflowType
    similarity_measure: SimilarityMeasure = SimilarityMeasure.STATEMENT
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE
    constrain_outputs: list[str] | None = None


class Config(BaseConfig):
    @classmethod
    def from_input(cls, input: ConfigInput, workflow_type: WorkflowType, model: str | None) -> "Config":
        defaults_for_quality = DEFAULT_CONFIG_FOR_QUALITY[input.quality_preset]
        defaults_for_workflow = DEFAULT_CONFIG_FOR_QUALITY_AND_WORKFLOW[input.quality_preset].get(
            workflow_type
        ) or DEFAULT_CONFIG_FOR_QUALITY_AND_WORKFLOW[input.quality_preset].get(WorkflowType.DEFAULT, {})
        reasoning_default = (
            ReasoningEffort.HIGH if workflow_type == WorkflowType.STRUCTURED_OUTPUT_SCORING else ReasoningEffort.NONE
        )
        params = {
            "reasoning_effort": reasoning_default,
            "use_prompt_evaluation": workflow_type == WorkflowType.RAG,
            "model": model or settings.DEFAULT_MODEL,
            **defaults_for_quality,
            **defaults_for_workflow,
            "similarity_measure": SimilarityMeasure.for_workflow(workflow_type),
            **input.model_dump(exclude_unset=True, exclude_none=True),
        }
        return cls(**params, workflow_type=workflow_type)
