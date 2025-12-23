import pytest

from tlm.utils.chat_completion_validation import _validate_chat_completion_params


def test_validate_chat_completion_params_allows_valid_openai_keys() -> None:
    params = {"messages": [], "model": "gpt-4.1", "temperature": 0.5}

    _validate_chat_completion_params(params)


def test_validate_chat_completion_params_allows_provider_as_none() -> None:
    params = {"messages": [], "model": "gpt-4.1", "temperature": 0.5}

    _validate_chat_completion_params(params)


def test_validate_chat_completion_params_requires_messages() -> None:
    params = {"model": "gpt-4.1-mini"}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params)

    assert "openai_args must include the following parameter(s): messages" in str(exc_info.value)
