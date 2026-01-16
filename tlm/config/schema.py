from tlm.config.presets import QualityPreset, ReasoningEffort
from tlm.types import SimilarityMeasure

from pydantic import BaseModel, Field


class ReferenceCompletionConfigSchema(BaseModel):
    num_reference_completions: int | None = Field(
        default=None, description="The attempted number of reference completions to generate."
    )


class ObservedConsistencyConfigSchema(BaseModel):
    num_consistency_completions: int | None = Field(
        default=None, description="The attempted number of observed consistency completions to generate."
    )
    observed_consistency_temperature: float | None = None


class SelfReflectionConfigSchema(BaseModel):
    self_reflection_temperature: float | None = None
    num_self_reflection_completions: int | None = Field(
        default=None,
        description=(
            "The number of self reflection prompts to use. Note that the first X number of prompts will be used, "
            "i.e. the order of the prompt templates in SELF_REFLECTION_TEMPLATES_BY_WORKFLOW[workflow_type] matters. "
            "-1 means all prompts will be used."
        ),
    )


class SemanticEvalsConfigSchema(BaseModel):
    use_prompt_evaluation: bool | None = None
    prompt_evaluation_temperature: float | None = None
    semantic_evaluation_temperature: float | None = None


class ModelProviderSchema(BaseModel):
    provider: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    api_version: str | None = None


class Config(
    ReferenceCompletionConfigSchema,
    ObservedConsistencyConfigSchema,
    SelfReflectionConfigSchema,
    SemanticEvalsConfigSchema,
    ModelProviderSchema,
):
    quality_preset: QualityPreset = QualityPreset.MEDIUM
    reasoning_effort: ReasoningEffort | None = None
    similarity_measure: SimilarityMeasure | None = None
    constrain_outputs: list[str] | None = None
