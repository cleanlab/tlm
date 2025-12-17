from typing import Dict, Any

from tlm.types import CompletionParams


def extract_user_prompt(completion_params: CompletionParams) -> str:
    messages = completion_params.get("messages")
    if messages is None:
        raise ValueError("messages are required in the completion params")

    # find the latest user message
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message["role"] == "user":
            return message["content"]

    raise ValueError("user prompt not found in the completion params")


def format_user_request(completions_params: CompletionParams) -> str:
    user_request = "Input Messages:\n"
    for message in completions_params["messages"]:
        user_request += f"{message['role']}: {message['content']}\n"

    user_request += get_response_format_prompt(completions_params.get("response_format", {}))

    return user_request


def get_response_format_prompt(response_format: Dict[str, Any]) -> str:
    if not response_format:
        return ""

    JSON_FORMAT_PROMPT = """\n\n## Response Format

The response must follow this JSON schema:

<json_schema>
{response_format}
</json_schema>\n"""

    try:
        extracted_response_format = response_format["json_schema"]["schema"]
    except KeyError:
        extracted_response_format = response_format

    return JSON_FORMAT_PROMPT.format(response_format=extracted_response_format)
