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
    """Strategies for scoring the similarity of two generated responses.

    Values:
        `JACCARD`, `EMBEDDING_SMALL`, `EMBEDDING_LARGE`, `CODE`, `STATEMENT`
    """

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
    """Criteria for performing a semantic evaluation of the query, context, and/or response.
    At least one of query_identifier, context_identifier, and response_identifier must be provided.

    Attributes:
        name: The name of the evaluation.
        criteria: Semantic description of the criteria to assess.
        query_identifier: Identifier for the user query to be provided in the prompt passed to the LLM, e.g. "User Query". Should be `None` if the evaluation does not require the query.
        context_identifier: Identifier for the context to be provided in the prompt passed to the LLM, e.g. "Context". Should be `None` if the evaluation does not require the context.
        response_identifier: Identifier for the response to be provided in the prompt passed to the LLM, e.g. "Response". Should be `None` if the evaluation does not require the response.
    """

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
