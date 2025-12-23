from pydantic import BaseModel, model_validator

from tlm.config.models import (
    AZURE_MODELS,
    BEDROCK_MODELS,
    BEDROCK_MODEL_TO_INFERENCE_PROFILE_ID,
    GOOGLE_MODELS,
    OPENAI_MODELS,
)


class APICredentials(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None


OPENAI_PROVIDER = "openai"
BEDROCK_PROVIDER = "bedrock"
GOOGLE_PROVIDER = "google"
AZURE_PROVIDER = "azure"


class ModelProvider(APICredentials):
    model: str
    provider: str | None = None

    @model_validator(mode="after")
    def set_provider_from_model(self):
        """Automatically set provider based on model name if provider is None."""
        if self.provider is None:
            if self.model in OPENAI_MODELS:
                self.provider = OPENAI_PROVIDER
            elif self.model in BEDROCK_MODELS:
                self.provider = BEDROCK_PROVIDER
            elif self.model in GOOGLE_MODELS:
                self.provider = GOOGLE_PROVIDER
            elif self.model in AZURE_MODELS:
                self.provider = AZURE_PROVIDER

        if self.model in BEDROCK_MODELS:
            self.model = BEDROCK_MODEL_TO_INFERENCE_PROFILE_ID[self.model]

        return self
