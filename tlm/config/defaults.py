from functools import lru_cache
from pathlib import Path
import logging
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tlm.config.models import DEFAULT_MODEL
from tlm.config.provider import ModelProvider

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory by looking for .env file."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / ".env").exists():
            return current
        current = current.parent
    # Fallback to current directory
    return Path.cwd()


class ProviderAuthSettings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str | None = "us-east-1"
    DEFAULT_API_KEY: str | None = None

    @model_validator(mode="after")
    def set_default_api_key(self):
        """Set DEFAULT_API_KEY to OPENAI_API_KEY if DEFAULT_API_KEY is not set."""
        if not self.DEFAULT_API_KEY:
            if self.DEFAULT_PROVIDER == "openai" and self.OPENAI_API_KEY:
                self.DEFAULT_API_KEY = self.OPENAI_API_KEY
            elif self.DEFAULT_PROVIDER == "gemini" and self.GEMINI_API_KEY:
                self.DEFAULT_API_KEY = self.GEMINI_API_KEY
            elif self.DEFAULT_PROVIDER == "deepseek" and self.DEEPSEEK_API_KEY:
                self.DEFAULT_API_KEY = self.DEEPSEEK_API_KEY

        if not self.DEFAULT_API_KEY:
            logger.warning("DEFAULT_API_KEY is not set")

        return self


class ModelSettings(BaseSettings):
    DEFAULT_MODEL: str = DEFAULT_MODEL
    DEFAULT_PROVIDER: str = "openai"
    TOP_LOGPROBS: int = 5


class TokenSettings(BaseSettings):
    MAX_TOKENS: int = 512
    EXPLANATION_LENGTH_UNDERESTIMATE_FACTOR: float = 0.8  # Percent at which to under-estimate the allowed length of explanation in observed consistency response to avoid sacrificing the answer via truncation. Smaller values of this factor lead to greater underestimates, set = 1.0 to avoid underestimating on purpose.
    AVG_WORDS_PER_TOKEN: float = 0.75  # OpenAI reported value of average number of words per token


class ScoreSettings(BaseSettings):
    SELF_REFLECTION_PARSE_FAILURE_SCORE: float = 0.5
    EXPLAINABILITY_THRESHOLD: float = 0.8
    SELF_REFLECTION_EXPLAINABILITY_THRESHOLD: float = 0.85
    CONSISTENCY_EXPLAINABILITY_THRESHOLD: float = 0.85


class Settings(
    ProviderAuthSettings,
    ModelSettings,
    TokenSettings,
    ScoreSettings,
):
    model_config = SettingsConfigDict(
        env_file=str(find_project_root() / ".env"), env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def default_model_provider(self) -> ModelProvider:
        return ModelProvider(model=self.DEFAULT_MODEL, api_key=self.DEFAULT_API_KEY, provider=self.DEFAULT_PROVIDER)


@lru_cache
def get_settings() -> Settings:
    return Settings()
