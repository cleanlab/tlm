import logging
import copy
import json
import string
from typing import Any, Dict
import re
from pydantic import BaseModel
from openai.lib._parsing._completions import type_to_response_format_param

import litellm
import litellm.exceptions
from litellm import Choices, acompletion
from litellm.files.main import ModelResponse
from litellm.types.utils import ChoiceLogprobs

from tlm.config.defaults import get_settings
from tlm.config.provider import ModelProvider
from tlm.types import (
    Completion,
    CompletionFailure,
    CompletionFailureType,
    CompletionParams,
    CompletionUsage,
    ExtractedResponseField,
    CompletionTemplate,
)
from tlm.utils.openai_utils import extract_structured_output_field, extract_message_content
from tlm.utils.constrain_outputs_utils import constrain_output
from tlm.utils.parse_utils import get_parsed_answer_tokens_confidence
from tlm.utils.scoring.per_field_scoring_utils import extract_per_field_reflection_metadata
from tlm.utils.math_utils import harmonic_mean

litellm.suppress_debug_info = True
litellm.set_verbose = False

settings = get_settings()

logger = logging.getLogger(__name__)
# TODO: decide how to handle logging in this library


async def generate_completion(
    template: CompletionTemplate,
    *,
    completion_params: CompletionParams = {},
    template_kwargs: dict[str, Any] = {},
    temperature: float | None = None,
    response_format_model: type[BaseModel] | None = None,
) -> Completion | CompletionFailure:
    litellm_params = _build_litellm_params(
        template,
        completion_params,
        template_kwargs,
        temperature,
        response_format_model,
    )

    completion = await _generate_completion(litellm_params, template)

    if isinstance(completion, Completion):
        log_msg = f"""Generated {template.__class__.__name__} completion for model {litellm_params["model"]} with messages:
    {json.dumps(litellm_params["messages"], indent=2)}

    Content:
    {completion.message}
    Explanation:
    {completion.explanation}
    Response fields:
    {json.dumps(completion.response_fields, indent=2)}
    ===============================================
    """
        print(log_msg)

    return completion


def _build_litellm_params(
    template: CompletionTemplate,
    completion_params: CompletionParams,
    template_kwargs: dict[str, Any] = {},
    temperature: float | None = None,
    response_format_model: type[BaseModel] | None = None,
) -> CompletionParams:
    litellm_params = copy.deepcopy(completion_params)

    input_messages: list[dict[str, str]] = litellm_params.get("messages", [])
    litellm_params["messages"] = template.format_messages(messages=input_messages, **template_kwargs)

    model = completion_params.get("model")

    model_provider = ModelProvider(model=model) if model else settings.default_model_provider
    litellm_params["model"] = model_provider.model

    if "max_tokens" not in litellm_params:
        litellm_params["max_tokens"] = settings.MAX_TOKENS

    if temperature:
        litellm_params["temperature"] = temperature

    overrides = template.get_completion_param_overrides(model_provider)
    litellm_params.update(overrides)

    if response_format_model:
        litellm_params["response_format"] = type_to_response_format_param(response_format_model)

    model_provider_params = model_provider.model_dump(exclude_unset=True, exclude_none=True)
    model_provider_params.pop("model", None)
    model_provider_params.pop("provider", None)

    litellm_params.update(model_provider_params)

    return litellm_params


async def _generate_completion(
    litellm_params: CompletionParams,
    template: CompletionTemplate | None,
) -> Completion | CompletionFailure:
    try:
        response = await acompletion(**litellm_params)
    except Exception as e:
        if isinstance(e, litellm.exceptions.Timeout):
            failure_type = CompletionFailureType.TIMEOUT
        elif isinstance(e, litellm.exceptions.APIError):
            failure_type = CompletionFailureType.API_ERROR
        else:
            failure_type = CompletionFailureType.RUNTIME_ERROR

        logger.error(
            f"[{template.__class__.__name__}] error generating completion with LiteLLM: {e}\nusing litellm params: \n{litellm_params}\n{'=' * 100}"
        )
        return CompletionFailure(type=failure_type, error=str(e))

    if isinstance(response, ModelResponse):
        assert isinstance(response.choices[0], Choices)
        content = response.choices[0].message.content or ""
        logprobs = None

        if litellm_params.get("logprobs") and hasattr(response.choices[0], "logprobs"):
            # Convert ChoiceLogprobs to dict to avoid Pydantic validation issues
            choice_logprobs = response.choices[0].logprobs
            logprobs = ChoiceLogprobs.model_validate(
                choice_logprobs.model_dump()
                if choice_logprobs and hasattr(choice_logprobs, "model_dump")
                else choice_logprobs
            )

            if raw_message_content := _get_raw_message_content(logprobs):
                content = raw_message_content

        explanation = (
            response.choices[0].message.reasoning_content
            if hasattr(response.choices[0].message, "reasoning_content")
            else None
        )

        usage = getattr(response, "usage", None)
        completion = Completion(
            message=content,
            explanation=explanation,
            logprobs=logprobs,
            usage=CompletionUsage(
                prompt_tokens=usage.prompt_tokens if usage else -1,
                completion_tokens=usage.completion_tokens if usage else -1,
                total_tokens=usage.total_tokens if usage else -1,
            ),
            original_response=response,
            template=template,
        )
        _parse_completion(completion)
        return completion

    print(f"unhandled response type: {type(response)}")
    return CompletionFailure(
        type=CompletionFailureType.RUNTIME_ERROR,
        error=f"unhandled response type: {type(response)}",
    )


def _get_raw_message_content(logprobs: ChoiceLogprobs) -> str | None:
    if logprobs is None or logprobs.content is None:
        return None
    return "".join([message_token.token for message_token in logprobs.content])


def _parse_completion(completion: Completion) -> None:
    """Update the completion with parsed response fields"""

    if not completion.template:
        return

    answer_start_idx, answer_end_idx = None, None

    for field, regex_patterns in completion.template.parse_patterns.items():
        match: re.Match[str] | None = None
        for regex_pattern in regex_patterns:
            pattern_strings = [regex_pattern.regex] if isinstance(regex_pattern.regex, str) else regex_pattern.regex
            for pattern_str in pattern_strings:
                match = re.search(pattern_str, completion.message, regex_pattern.flags)
                if match:
                    group_idx = 1
                    field_value = match.group(group_idx).strip()
                    completion.add_response_field(field, field_value)

                    if field == ExtractedResponseField.ANSWER:
                        answer_start_idx = match.start(group_idx)
                        answer_end_idx = match.end(group_idx)

                    if field == ExtractedResponseField.SCORE:
                        if score_mapper := completion.template.score_mapper:
                            completion.add_response_field(
                                ExtractedResponseField.MAPPED_SCORE, score_mapper(field_value)
                            )
                    break
            if match:
                break

    if completion.template.constrain_outputs:
        constrain_output(completion, completion.message, completion.template.constrain_outputs)

    if answer_start_idx and answer_end_idx and completion.logprobs:
        answer_end_idx = _get_trimmed_index(completion.message, answer_start_idx, answer_end_idx)
        generic_answer_tokens_confidence = get_parsed_answer_tokens_confidence(
            completion,
            answer_start_idx,
            answer_end_idx,
        )
        completion.perplexity = generic_answer_tokens_confidence

    if completion.template.extract_answer:
        message_content = (
            extract_message_content(completion.original_response)
            if isinstance(completion.original_response, Dict)
            else completion.message
        )
        answer = extract_structured_output_field(message_content, "answer")
        completion.add_response_field(ExtractedResponseField.ANSWER, answer or message_content)

        explanation = extract_structured_output_field(message_content, "explanation")
        completion.add_response_field(ExtractedResponseField.EXPLANATION, explanation)

    if completion.template.per_field_score_key:
        if isinstance(completion.original_response, Dict):
            message_content = extract_message_content(completion.original_response)
        else:
            message_content = completion.message

        assert completion.template.score_mapper is not None
        per_field_metadata = extract_per_field_reflection_metadata(
            message_content, completion.template.per_field_score_key, completion.template.score_mapper
        )
        completion.per_field_metadata = per_field_metadata
        harmonic_mean_score = harmonic_mean([metadata.score for metadata in per_field_metadata.values()])
        completion.add_response_field(ExtractedResponseField.MAPPED_SCORE, harmonic_mean_score)


def _get_trimmed_index(message: str, start_idx: int, end_idx: int) -> int:
    """Returns an adjusted end index that excludes any trailing punctuation and whitespace.

    Args:
        message: The full message string
        start_idx: The start index of the matched text
        end_idx: The end index of the matched text

    Returns:
        The adjusted end index with trailing punctuation and preceding whitespace excluded
    """
    for i in range(end_idx - 1, start_idx - 1, -1):
        char = message[i]
        if not char.isspace() and char not in string.punctuation:
            return i + 1

    return start_idx


def get_cleaned_chat_completion(completion: Completion) -> Dict[str, Any] | ModelResponse:
    if isinstance(completion.original_response, Dict):
        return completion.original_response
    else:
        answer_only = completion.response_fields.get(ExtractedResponseField.ANSWER)
        if not answer_only:
            return completion.original_response

        cleaned_response = completion.original_response.model_copy(deep=True)
        # amend the last assistant message to remove the reasoning, if present
        if choices := cleaned_response.get("choices", []):
            if last_choice := choices[-1]:
                if last_choice.get("message"):
                    if last_choice["message"].get("role") == "assistant" and last_choice["message"].get("content"):
                        last_choice["message"]["content"] = answer_only

        return cleaned_response
