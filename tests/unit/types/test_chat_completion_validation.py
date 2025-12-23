import pytest

from tlm.utils.chat_completion_validation import _validate_chat_completion_params


def test_validate_chat_completion_params_allows_valid_openai_keys() -> None:
    params = {"messages": [], "model": "gpt-4.1", "temperature": 0.5}

    _validate_chat_completion_params(params, "openai")


def test_validate_chat_completion_params_allows_provider_as_none() -> None:
    params = {"messages": [], "model": "gpt-4.1", "temperature": 0.5}

    _validate_chat_completion_params(params, None)


def test_validate_chat_completion_params_requires_messages() -> None:
    params = {"model": "gpt-4.1-mini"}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params, "openai")

    assert "openai_args must include the following parameter(s): messages" in str(exc_info.value)


def test_validate_chat_completion_params_flags_invalid_keys() -> None:
    params = {"messages": [], "model": "gpt-4.1", "bad": True}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params, "openai")

    assert "Unsupported chat completion parameter(s) for provider openai: bad" in str(exc_info.value)


def test_validate_chat_completion_params_allows_bedrock_only_keys() -> None:
    params = {"messages": [], "system": "Be helpful"}

    _validate_chat_completion_params(params, "bedrock")

    with pytest.raises(ValueError):
        _validate_chat_completion_params(params, "openai")
