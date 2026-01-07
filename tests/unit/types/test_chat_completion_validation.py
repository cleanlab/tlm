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


def test_validate_chat_completion_params_requires_messages_list() -> None:
    params = {"messages": "not-a-list"}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params)

    assert "`messages` must be provided as a list" in str(exc_info.value)


def test_validate_chat_completion_params_requires_message_dict() -> None:
    params = {"messages": ["not-a-dict"]}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params)

    assert "messages[0] must be a dictionary" in str(exc_info.value)


def test_validate_chat_completion_params_requires_role_and_content_strings() -> None:
    params = {"messages": [{"role": 123, "content": None}]}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params)

    assert "messages[0]['role']" in str(exc_info.value)


def test_validate_chat_completion_params_allows_function_call_without_content() -> None:
    params = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "foo", "arguments": '{"bar": 1}'},
            }
        ]
    }

    _validate_chat_completion_params(params)


def test_validate_chat_completion_params_requires_content_when_no_function_call() -> None:
    params = {"messages": [{"role": "assistant", "content": None}]}

    with pytest.raises(ValueError) as exc_info:
        _validate_chat_completion_params(params)

    assert "messages[0]['content'] must be a string." in str(exc_info.value)
