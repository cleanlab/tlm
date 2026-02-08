from typing import Literal

from pydantic import BaseModel, Field


class PerFieldScoreEvaluationBase(BaseModel):
    explanation: str


class PerFieldCertaintyEvaluation(PerFieldScoreEvaluationBase):
    score: int = Field(ge=0, le=100)


class PerFieldCorrectnessEvaluation(PerFieldScoreEvaluationBase):
    confidence: Literal["Certain", "Mostly Certain", "Somewhat Certain", "Uncertain", "Likely Incorrect"]


# base class type for incorrect field evaluation
class IncorrectFieldEvaluationBase(BaseModel): ...
