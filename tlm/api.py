from typing import Any, cast

import asyncio
import sys
from openai.types.chat import ChatCompletion

from tlm.config.base import Config, ConfigInput
from tlm.config.presets import WorkflowType
from tlm.inference import InferenceResult, tlm_inference
from tlm.types import SemanticEval


def is_notebook() -> bool:
    """Returns True if running in a notebook, False otherwise."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        return bool("IPKernelApp" in get_ipython().config)
    except Exception:
        return False


class TLM:
    def __init__(
        self,
        config_input: ConfigInput = ConfigInput(),
        evals: list[SemanticEval] | None = None,
    ):
        self.config_input = config_input
        self.evals = evals

        is_notebook_flag = is_notebook()

        if is_notebook_flag:
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()

        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()

    def create(
        self,
        *,
        context: str | None = None,
        evals: list[SemanticEval] | None = None,
        **openai_kwargs: Any,
    ) -> InferenceResult:
        return self._event_loop.run_until_complete(
            self._async_inference(
                context=context,
                evals=evals,
                **openai_kwargs,
            )
        )

    def score(
        self,
        *,
        response: ChatCompletion | dict[str, Any],
        context: str | None = None,
        evals: list[SemanticEval] | None = None,
        **openai_kwargs: Any,
    ) -> InferenceResult:
        if isinstance(response, ChatCompletion):
            response = cast(dict[str, Any], response.model_dump())

        return self._event_loop.run_until_complete(
            self._async_inference(
                response=response,
                context=context,
                evals=evals,
                **openai_kwargs,
            )
        )

    async def _async_inference(
        self,
        *,
        response: dict[str, Any] | None = None,
        context: str | None = None,
        evals: list[SemanticEval] | None = None,
        **openai_kwargs: Any,
    ) -> InferenceResult:
        workflow_type = WorkflowType.from_inference_params(
            openai_args=openai_kwargs,
            score=response is not None,
            rag=(context is not None),
            constrain_outputs=self.config_input.constrain_outputs,
        )
        model = openai_kwargs.get("model")
        config = Config.from_input(self.config_input, workflow_type, model)
        return await tlm_inference(
            completion_params=openai_kwargs,
            response=response,
            evals=evals,
            context=context,
            config=config,
        )
