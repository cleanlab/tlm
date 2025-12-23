"""Validation helpers for chat completion parameter dictionaries."""

from __future__ import annotations
from tlm.types.base import CompletionParams
from tlm.config.provider import AZURE_PROVIDER, BEDROCK_PROVIDER, GOOGLE_PROVIDER, OPENAI_PROVIDER
from typing import FrozenSet

VALID_OPENAI_CHAT_COMPLETION_PARAMS: FrozenSet[str] = frozenset(
    {
        "audio",
        "function_call",
        "functions",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
)
VALID_AZURE_CHAT_COMPLETION_PARAMS: FrozenSet[str] = VALID_OPENAI_CHAT_COMPLETION_PARAMS
VALID_GOOGLE_CHAT_COMPLETION_PARAMS: FrozenSet[str] = VALID_OPENAI_CHAT_COMPLETION_PARAMS
VALID_BEDROCK_CHAT_COMPLETION_PARAMS: FrozenSet[str] = frozenset(
    {
        *VALID_OPENAI_CHAT_COMPLETION_PARAMS,
        "betas",
        "container",
        "context_management",
        "mcp_servers",
        "output_config",
        "output_format",
        "stop_sequences",
        "system",
        "thinking",
        "top_k",
        "top_p",
        "tool_config",
    }
)

def _resolve_valid_chat_completion_params(provider: str | None) -> FrozenSet[str]:
    if provider == OPENAI_PROVIDER:
        return VALID_OPENAI_CHAT_COMPLETION_PARAMS
    elif provider == BEDROCK_PROVIDER:
        return VALID_BEDROCK_CHAT_COMPLETION_PARAMS
    elif provider == GOOGLE_PROVIDER:
        return VALID_GOOGLE_CHAT_COMPLETION_PARAMS
    elif provider == AZURE_PROVIDER:
        return VALID_AZURE_CHAT_COMPLETION_PARAMS
    else:
        return VALID_OPENAI_CHAT_COMPLETION_PARAMS


REQUIRED_CHAT_COMPLETION_PARAMS: FrozenSet[str] = frozenset({"messages"})


def _validate_chat_completion_params(params: CompletionParams, provider: str | None) -> None:
    """Ensure only supported chat completion params are passed into inference."""

    missing_required = [param for param in REQUIRED_CHAT_COMPLETION_PARAMS if param not in params]
    if missing_required:
        required_str = ", ".join(sorted(REQUIRED_CHAT_COMPLETION_PARAMS))
        raise ValueError(f"openai_args must include the following parameter(s): {required_str}")

    valid_params = _resolve_valid_chat_completion_params(provider)

    invalid_keys = sorted(set(params.keys()) - valid_params)
    if invalid_keys:
        raise ValueError(
            f"Unsupported chat completion parameter(s) for provider {provider}: "
            + ", ".join(invalid_keys)
        )
