from tlm.config.presets import QualityPreset, ReasoningEffort
from tlm.types import SimilarityMeasure

from pydantic import BaseModel, Field


class ReferenceCompletionConfigSchema(BaseModel):
    """
    Configuration for reference completion generation.

    Attributes:
        num_reference_completions: The attempted number of reference completions to generate.
    """

    num_reference_completions: int | None = Field(
        default=None, description="The attempted number of reference completions to generate."
    )


class ObservedConsistencyConfigSchema(BaseModel):
    """
    Configuration for generating additional completions against which to score consistency of reference completions.

    Attributes:
        num_consistency_completions: The attempted number of observed consistency completions to generate.
        observed_consistency_temperature: The temperature to use for generating comparison completions.
    """

    num_consistency_completions: int | None = Field(
        default=None, description="The attempted number of observed consistency completions to generate."
    )
    observed_consistency_temperature: float | None = None


class SelfReflectionConfigSchema(BaseModel):
    """
    Configuration for prompting LLM-as-judge to score the trustworthiness of reference completions using self-reflection prompts.

    Attributes:
        self_reflection_temperature: The temperature to use for self reflection completions.
        num_self_reflection_completions: The attempted number of self reflection completions to generate.
    """

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
    """
    Configuration for semantic evaluation of reference completions.

    Attributes:
        use_prompt_evaluation: Whether to incorporate prompt evaluation scores into the final trustworthiness score.
        prompt_evaluation_temperature: The temperature to use for prompt evaluation completions.
        semantic_evaluation_temperature: The temperature to use when generating completions to score the Evals.
    """

    use_prompt_evaluation: bool | None = None
    prompt_evaluation_temperature: float | None = None  # TODO: rename to prompt_evaluation_temperature
    semantic_evaluation_temperature: float | None = None  # TODO: rename to semantic_evaluation_temperature


class ModelProviderSchema(BaseModel):
    """
    Configuration for the model provider in alignment with the LiteLLM API.

    Attributes:
        provider: The name of the model provider.
        api_base: The base URL of the model provider's API.
        api_key: The API key to use for the model provider.
        api_version: The version of the model provider's API.
    """

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
    """Configuration for TLM inference.

    This class combines multiple configuration schemas to provide comprehensive
    control over TLM's inference behavior, including reference completions,
    consistency checking, self-reflection, semantic evaluation, and model provider settings.

    Attributes:
        quality_preset: Quality preset controlling the trade-off between speed and accuracy.
        reasoning_effort: Optional reasoning effort level for models that support it.
        similarity_measure: Optional similarity measure to use for comparing consistency across responses.
        constrain_outputs: Optional list of allowed output values to constrain responses, for example in multiple choice questions.
    """

    quality_preset: QualityPreset = QualityPreset.MEDIUM
    reasoning_effort: ReasoningEffort | None = None
    similarity_measure: SimilarityMeasure | None = None
    constrain_outputs: list[str] | None = None
