from typing import Any

from tlm.config.base import Config, ConfigInput
from tlm.config.presets import WorkflowType
from tlm.inference import InferenceResult, tlm_inference
from tlm.types import SemanticEval, CompletionParams
from tlm.utils.chat_completion_validation import _validate_chat_completion_params


async def inference(
    *,
    openai_args: CompletionParams,
    response: dict[str, Any] | None = None,  # TODO: support passing multiple reference completions?
    context: str | None = None,
    evals: list[SemanticEval] | None = None,
    config_input: ConfigInput = ConfigInput(),
) -> InferenceResult:
    _validate_chat_completion_params(openai_args)
    workflow_type = WorkflowType.from_inference_params(
        openai_args=openai_args,
        score=response is not None,
        rag=(context is not None),
        constrain_outputs=config_input.constrain_outputs,
    )
    config = Config.from_input(config_input, workflow_type)
    return await tlm_inference(
        completion_params=openai_args,
        response=response,
        evals=evals,
        context=context,
        config=config,
    )
