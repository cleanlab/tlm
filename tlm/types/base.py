from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel
import re

from tlm.config.presets import WorkflowType


class InferenceType(str, Enum):
    SCORE = "score"
    PROMPT = "prompt"


# TODO: just convert these to properties on the Completion object
class ExtractedResponseField(str, Enum):
    MESSAGE = "response"
    ANSWER = "answer"
    EXPLANATION = "reasoning"
    SCORE = "score"
    MAPPED_SCORE = "mapped_score"


class SimilarityMeasure(str, Enum):
    JACCARD = "jaccard"  # formerly STRING
    EMBEDDING_SMALL = "embedding_small"
    EMBEDDING_LARGE = "embedding_large"
    CODE = "code"
    STATEMENT = "statement"  # formerly DISCREPANCY

    @classmethod
    def for_workflow(cls, workflow_type: WorkflowType) -> "SimilarityMeasure":
        if workflow_type == WorkflowType.QA:
            return cls.STATEMENT
        elif workflow_type == WorkflowType.CLASSIFICATION:
            return cls.EMBEDDING_SMALL
        elif workflow_type == WorkflowType.BINARY_CLASSIFICATION:
            return cls.EMBEDDING_LARGE
        elif workflow_type == WorkflowType.RAG:
            return cls.CODE
        elif workflow_type == WorkflowType.STRUCTURED_OUTPUT_SCORING:
            return cls.JACCARD

        return cls.STATEMENT  # default


class CompletionFailureType(Enum):
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    PARSE = "parse"


class FieldMetadata(BaseModel):
    score: float
    explanation: str


class Eval(BaseModel):
    name: str
    criteria: str
    query_identifier: str | None = None
    context_identifier: str | None = None
    response_identifier: str | None = None


class RegexPattern(BaseModel):
    regex: str | list[str]
    flags: int = re.DOTALL


class AnswerChoiceToken(BaseModel):
    token: str
    positive: bool


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionFailure(BaseModel):
    error: str | None = None
    type: CompletionFailureType | None = None


CompletionParams = Dict[str, Any]
