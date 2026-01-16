from typing import Any

import asyncio
import sys
from openai.types.chat import ChatCompletion

from tlm.config.base import BaseConfig
from tlm.config.schema import Config
from tlm.config.presets import WorkflowType
from tlm.inference import InferenceResult, tlm_inference
from tlm.types import Eval


def is_notebook() -> bool:
    """Returns True if running in a notebook, False otherwise."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        return bool("IPKernelApp" in get_ipython().config)
    except Exception:
        return False


class TLM:
    """Trustworthy Language Model (TLM) for scoring the trustworthines of responses from any LLM in real-time.

    TLM can either score responses that you have already generated from your own LLM,
    or simultaneously generate responses and score their trustworthiness.
    """

    def __init__(
        self,
        config: Config = Config(),
        evals: list[Eval] | None = None,
    ):
        """Initialize a TLM instance.

        Args:
            config_input: Configuration for TLM behavior. Controls quality presets,
                reasoning effort, number of reference/consistency completions, and
                other component settings. Defaults to medium quality preset.
            evals: Optional list of evaluations. Each evaluation
                defines a name, criteria, and optional query/context/response identifiers.
        """
        self.config = config
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
        evals: list[Eval] | None = None,
        **openai_kwargs: Any,
    ) -> InferenceResult:
        """Create a new LLM completion and then score its trustworthiness.

        This method generates a completion using the provided OpenAI-API-compatible arguments,
        and then runs TLM to produce a trustworthiness score for this completion.

        Args:
            context: Optional context string for RAG workflows. When provided, enables
                RAG-specific evaluations and prompt evaluation.
            evals: Optional list of semantic evaluations to apply. Overrides any
                evaluations provided during TLM initialization.
            **openai_kwargs: OpenAI-compatible completion parameters. Common parameters
                include:
                - messages: List of message dicts with "role" and "content" keys
                - model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
                - temperature: Sampling temperature
                - response_format: For structured outputs (e.g., JSON schema)
                - Any other OpenAI API parameters

        Returns:
            InferenceResult object containing:
                - response: The generated response (string or dict for structured outputs)
                - trustworthiness_score: Trustworthiness score between 0 and 1
                - usage: Token usage information
                - metadata: Additional metadata (e.g., per-field scores for structured outputs)
                - evals: Dictionary of additional evaluation scores (if evals are provided)
                - explanation: Optional explanation for the trustworthiness score
        """
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
        evals: list[Eval] | None = None,
        **openai_kwargs: Any,
    ) -> InferenceResult:
        """Score the trusworthiness of an existing LLM response/completion (from any LLM, or even from a human-writer).

        Args:
            response: The existing response to score. Can be either an OpenAI
                ChatCompletion object or a dictionary representation of a chat completion.
            context: Optional context string for RAG workflows. When provided, enables
                RAG-specific evaluations.
            evals: Optional list of semantic evaluations to apply. Overrides any
                evaluations provxided during TLM initialization.
            **openai_kwargs: Optional OpenAI-compatible parameters. These are used for
                workflow type detection and configuration, but no new completion is
                generated.

        Returns:
            InferenceResult containing:
                - response: The original response (preserved from input)
                - trustworthiness_score: Trustworthiness score between 0 and 1
                - usage: Token usage information
                - metadata: Additional metadata (e.g., per-field scores for structured outputs)
                - evals: Dictionary of additional evaluation scores (if evals are provided)
                - explanation: Optional explanation for the trustworthiness score
        """
        if isinstance(response, ChatCompletion):
            response = {"chat_completion": response.model_dump()}

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
        evals: list[Eval] | None = None,
        **openai_kwargs: Any,
    ) -> InferenceResult:
        """Internal async method that performs the inference or scoring operation.

        This method handles workflow type detection, configuration creation, and
        delegates to the TLM inference pipeline. It is called by both `create()` and
        `score()` methods.
        """
        workflow_type = WorkflowType.from_inference_params(
            openai_args=openai_kwargs,
            score=response is not None,
            rag=(context is not None),
            constrain_outputs=self.config.constrain_outputs,
        )
        model = openai_kwargs.get("model")
        config = BaseConfig.from_input(self.config, workflow_type, model)
        return await tlm_inference(
            completion_params=openai_kwargs,
            response=response,
            evals=evals,
            context=context,
            config=config,
        )
