from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict
from litellm.types.utils import ChoiceLogprobs
from litellm.files.main import ModelResponse

from tlm.utils.openai_utils import extract_message_content
from .base import CompletionUsage, ExtractedResponseField, FieldMetadata
from .completion_template import CompletionTemplate


class Completion(BaseModel):
    message: str
    logprobs: ChoiceLogprobs | None = None
    perplexity: float | None = None
    usage: CompletionUsage | None = None
    explanation: str | None = None
    per_field_metadata: dict[str, FieldMetadata] | None = None
    response_fields: dict[str, Any] = Field(default_factory=dict)
    original_response: Dict[str, Any] | ModelResponse
    template: CompletionTemplate | None

    @field_validator("message", "explanation", mode="after")
    @classmethod
    def strip_whitespace(cls, v: str | None) -> str | None:
        if v is None:
            return None

        return v.strip()

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "Completion":
        response_str = response["response"]
        completion = cls(
            message=response_str, original_response=response, template=None, perplexity=response.get("perplexity")
        )
        completion.add_response_field(ExtractedResponseField.MESSAGE, response_str)
        completion.add_response_field(ExtractedResponseField.ANSWER, response_str)
        return completion

    @classmethod
    def from_completion_dict(cls, completion_dict: Dict[str, Any]) -> "Completion":
        message = extract_message_content(completion_dict)
        completion = cls(
            message=message,
            original_response=completion_dict,
            template=None,
            perplexity=completion_dict.get("perplexity"),
        )
        completion.add_response_field(ExtractedResponseField.ANSWER, message)
        return completion

    def add_response_field(self, field: ExtractedResponseField, value: Any):
        stripped_value = value.strip() if isinstance(value, str) else value

        self.response_fields[field.value] = stripped_value

        if field == ExtractedResponseField.MESSAGE:
            self.message = stripped_value
        if field == ExtractedResponseField.EXPLANATION:
            self.explanation = stripped_value
