from typing import Any, TypedDict

from tlm.config.base import Config
from tlm.config.presets import WorkflowType
from tlm.pipeline import PipelineFactory
from tlm.types import SemanticEval, CompletionParams
from tlm.utils.scoring.semantic_evaluation_scoring_utils import DEFAULT_RAG_EVALS


class InferenceResult(TypedDict):
    response: str | dict[str, Any]  # either a response string or OpenAI chat completion dict
    trustworthiness_score: float
    usage: dict[str, Any]
    metadata: dict[str, Any] | None
    evals: dict[str, float] | None
    explanation: str | None


async def tlm_inference(
    *,
    completion_params: CompletionParams,
    response: dict[str, Any] | None,
    evals: list[SemanticEval] | None,
    context: str | None,
    config: Config,
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
