"""Validation helpers for chat completion parameter dictionaries."""

from __future__ import annotations

# from litellm import get_supported_openai_params
from tlm.types.base import CompletionParams
from typing import FrozenSet


REQUIRED_CHAT_COMPLETION_PARAMS: FrozenSet[str] = frozenset({"messages"})


def _validate_chat_completion_params(params: CompletionParams) -> None:
    """Ensure only supported chat completion params are passed into inference."""

    missing_required = [param for param in REQUIRED_CHAT_COMPLETION_PARAMS if param not in params]
    if missing_required:
        required_str = ", ".join(sorted(REQUIRED_CHAT_COMPLETION_PARAMS))
        raise ValueError(f"openai_args must include the following parameter(s): {required_str}")

    return
