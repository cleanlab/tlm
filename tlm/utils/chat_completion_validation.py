"""Validation helpers for chat completion parameter dictionaries."""

from typing import Any, Callable, FrozenSet, Mapping

from tlm.types.base import CompletionParams


ParamValidator = Callable[[Any], None]

REQUIRED_CHAT_COMPLETION_PARAMS: FrozenSet[str] = frozenset({"messages"})


def _validate_messages_param(messages: Any) -> None:
    """Validate the shape of a `messages` param for chat completions."""

    if not isinstance(messages, list):
        raise ValueError("`messages` must be provided as a list of message dictionaries.")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be a dictionary.")

        role = message.get("role")
        content = message.get("content")

        if role is None or not isinstance(role, str):
            raise ValueError(f"messages[{index}]['role'] must be a non-empty string.")

        if content is None or not isinstance(content, str):
            function_call = message.get("function_call")
            if role != "assistant":
                raise ValueError(f"Non-assistant message at index {index} must have content.")
            if function_call is None:
                raise ValueError(f"Assistant message at index {index} must have content or a function call.")


REQUIRED_PARAM_VALIDATORS: Mapping[str, ParamValidator] = {
    "messages": _validate_messages_param,
}


def _validate_chat_completion_params(params: CompletionParams) -> None:  # type: ignore
    """Ensure only supported chat completion params are passed into inference."""

    missing_required = [param for param in REQUIRED_CHAT_COMPLETION_PARAMS if param not in params]
    if missing_required:
        required_str = ", ".join(sorted(REQUIRED_CHAT_COMPLETION_PARAMS))
        raise ValueError(f"openai_args must include the following parameter(s): {required_str}")

    for param in REQUIRED_CHAT_COMPLETION_PARAMS:
        validator = REQUIRED_PARAM_VALIDATORS.get(param)
        if validator is None:
            continue
        validator(params[param])
