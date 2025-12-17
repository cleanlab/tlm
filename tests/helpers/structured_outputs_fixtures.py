from pydantic import BaseModel
import pytest
from typing import Any


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


@pytest.fixture
def structured_outputs_completion_params() -> dict[str, Any]:
    openai_args = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "CalendarEvent",
                "strict": True,
                "schema": {
                    **CalendarEvent.model_json_schema(),
                    "additionalProperties": False,
                },
            },
        },
    }
    return openai_args


@pytest.fixture
def structured_outputs_reference_completion() -> dict[str, Any]:
    original_response = {
        "chat_completion": {
            "id": "chatcmpl-CZ25kwHT0DLqr4sLavIhFQ6BoGnfY",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": '{"name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}',
                        "refusal": None,
                        "role": "assistant",
                        "annotations": [],
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                        "parsed": None,
                    },
                }
            ],
            "created": 1762465556,
            "model": "gpt-4.1-mini-2025-04-14",
            "object": "chat.completion",
            "service_tier": "default",
            "system_fingerprint": "fp_4c2851f862",
            "usage": {
                "completion_tokens": 17,
                "prompt_tokens": 74,
                "total_tokens": 91,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
        },
        "perplexity": 0.95,
    }
    return original_response
