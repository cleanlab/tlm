from pydantic import BaseModel, Field
from typing import Any, Callable
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

from .base import ExtractedResponseField, RegexPattern, AnswerChoiceToken, CompletionParams

from tlm.config.models import MODELS_WITH_LOGPROBS
from tlm.config.provider import ModelProvider
from tlm.config.defaults import get_settings


settings = get_settings()


class CompletionTemplate(BaseModel):
    prompt_template: str | None = Field(
        description="The format string used for prompting the LLM, to be called using the kwargs and overrides"
    )
    parse_patterns: dict[ExtractedResponseField, list[RegexPattern]] = Field(
        default={},
        description="Mapping from response field to regex patterns that can be used to extract the field from the completion message",
    )
    answer_choice_tokens: list[AnswerChoiceToken] = Field(
        default=[], description="All possible answer choices that are expected based on the prompt"
    )
    score_mapper: Callable[[str], float] | None = Field(
        default=None, description="A function to map the parsed answer to a score"
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature used to generate the completion. If not provided, the LiteLLM default is 1.0",
    )
    per_field_score_key: str | None = Field(
        default=None,
        description="Key in each field of the completion response schema that contains the reflection score",
    )
    use_logprobs: bool | None = Field(
        default=None,
        description="Whether to use logprobs to generate the completion, if available",
    )
    constrain_outputs: list[str] | None = Field(
        default=None,
        description="List of outputs that the answer should be constrained to",
    )
    include_message_context: bool = Field(
        default=True,
        description="Whether to include multi-turn message context when generating the completion",
    )
    extract_answer: bool = Field(
        default=False,
        description="True indicates that the answer should be extracted from the 'answer' field of the structured output response",
    )

    @classmethod
    def construct_response_format(cls, response_json: str) -> type[BaseModel] | None:
        return None

    def get_completion_param_overrides(self, model_provider: ModelProvider) -> CompletionParams:
        overrides: CompletionParams = {}
        if self.temperature is not None:
            overrides["temperature"] = self.temperature
        if self.use_logprobs is False:
            overrides["logprobs"] = None
            overrides["top_logprobs"] = None
        elif self.use_logprobs is True:
            logprobs_supported = model_provider.model in MODELS_WITH_LOGPROBS
            top_logprobs_override = None

            if logprobs_supported:
                overrides["logprobs"] = True
                model_supported_params = get_supported_openai_params(
                    model=model_provider.model, custom_llm_provider=model_provider.provider
                )
                if model_supported_params and "top_logprobs" in model_supported_params:
                    top_logprobs_override = settings.TOP_LOGPROBS
            else:
                overrides["logprobs"] = False

            overrides["top_logprobs"] = top_logprobs_override

        return overrides

    def format_messages(
        self,
        messages: list[dict[str, str]] | None = None,
        **template_kwargs: Any,
    ) -> list[dict[str, str]]:
        if messages is None:
            messages = []
        else:
            messages = messages.copy()

        if self.prompt_template is None:
            return messages

        if self.include_message_context:
            # remove all trailing user messages
            while len(messages) > 0 and messages[-1]["role"] == "user":
                messages.pop(-1)
        else:
            messages = []

        formatted_prompt = self.prompt_template.format(**template_kwargs)
        messages.append({"role": "user", "content": formatted_prompt})
        return messages
