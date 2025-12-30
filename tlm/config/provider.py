from pydantic import BaseModel, model_validator

from tlm.config.models import (
    AZURE_MODELS,
    BEDROCK_MODELS,
    BEDROCK_MODEL_TO_INFERENCE_PROFILE_ID,
    GOOGLE_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
)


class APICredentials(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None


class ModelProvider(APICredentials):
    model: str
    provider: str | None = None

    @model_validator(mode="after")
    def set_provider_from_model(self):
        """Automatically set provider based on model name if provider is None."""
        if self.provider is None:
            if self.model in OPENAI_MODELS:
                self.provider = "openai"
            elif self.model in BEDROCK_MODELS:
                self.provider = "bedrock"
            elif self.model in GOOGLE_MODELS:
                # For Google models, the provider is already in the prefix
                # vertex_ai/ prefix = Vertex AI (uses ADC)
                # gemini/ prefix = Google AI Studio (uses API key)
                # The prefix is part of the model name, so provider is set by LiteLLM
                pass
            elif self.model in AZURE_MODELS:
                self.provider = "azure"
            elif self.model in DEEPSEEK_MODELS:
                self.provider = "deepseek"

        if self.model in BEDROCK_MODELS:
            self.model = BEDROCK_MODEL_TO_INFERENCE_PROFILE_ID[self.model]

        return self
