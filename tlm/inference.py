from typing import Any, TypedDict

from tlm.config.base import BaseConfig
from tlm.config.presets import WorkflowType
from tlm.pipeline import PipelineFactory
from tlm.types import Eval, CompletionParams
from tlm.utils.scoring.semantic_evaluation_scoring_utils import DEFAULT_RAG_EVALS


class InferenceResult(TypedDict):
    """Result returned from TLM inference.

    Attributes:
        response: Either a response string or dictionary representation of an OpenAI chat completion.
        trustworthiness_score: Score indicating the trustworthiness of the response, between 0 and 1.
        usage: Token usage information for the inference, including prompt and completion tokens.
        metadata: Optional metadata, e.g. per-field scores for structured outputs.
        evals: Optional dictionary of Eval scores, keyed by evaluation name.
        explanation: Explanation for the trustworthiness score.
    """

    response: str | dict[str, Any]
    trustworthiness_score: float
    usage: dict[str, Any]
    metadata: dict[str, Any] | None
    evals: dict[str, float] | None
    explanation: str | None


async def tlm_inference(
    *,
    completion_params: CompletionParams,
    response: dict[str, Any] | None,
    evals: list[Eval] | None,
    context: str | None,
    config: BaseConfig,
) -> InferenceResult:
    if evals is None and config.workflow_type == WorkflowType.RAG:
        evals = DEFAULT_RAG_EVALS

    pipeline = PipelineFactory.create(
        config=config,
        completion_params=completion_params,
        response=response,
        evals=evals,
        context=context,
    )
    results = await pipeline.run()

    best_response = results["best_response"]
    trustworthiness_score = results["trustworthiness_score"]
    usage = results.get("usage", {})
    explanation = results.get("explanation")
    evals_not_requiring_response: dict[str, float] = results.get("evals_not_requiring_response", {})
    evals_requiring_response: dict[str, float] = results.get("evals_requiring_response", {})
    metadata = {}
    if results.get("self_reflection_metadata_per_field"):
        metadata["per_field_score"] = results.get("self_reflection_metadata_per_field")

    return InferenceResult(
        response=best_response,
        trustworthiness_score=trustworthiness_score,
        usage=usage,
        metadata=metadata,
        evals={
            **evals_not_requiring_response,
            **evals_requiring_response,
        },
        explanation=explanation,
    )
