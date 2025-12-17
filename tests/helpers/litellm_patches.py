"""Shared test helper functions."""

from contextlib import contextmanager
from unittest.mock import patch

import tiktoken
import litellm
from litellm.files.main import ModelResponse
from litellm.types.utils import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob


def _generate_logprobs_from_text(text: str, encoding_name: str = "o200k_base") -> list[ChatCompletionTokenLogprob]:
    """Generate realistic logprobs from text using tiktoken for accurate tokenization.

    Args:
        text: The text to tokenize
        encoding_name: The tiktoken encoding to use (default: "o200k_base" for GPT-4.1+ models)

    Returns:
        List of ChatCompletionTokenLogprob objects matching the tokenized text
    """
    if not text:
        return []

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_ids = encoding.encode(text, disallowed_special=())

    tokens: list[tuple[str, list[int]]] = []
    for token_id in token_ids:
        token_bytes_list: list[int]
        try:
            token_bytes_obj = encoding.decode_single_token_bytes(token_id)
            token_str = token_bytes_obj.decode("utf-8", errors="replace")
            token_bytes_list = list(token_bytes_obj)
        except Exception:
            token_str = encoding.decode([token_id])
            token_bytes_list = list(token_str.encode("utf-8"))
        tokens.append((token_str, token_bytes_list))

    if not tokens:
        return []

    token_logprobs = []
    for i, (token, token_bytes) in enumerate(tokens):
        # Create realistic logprob values (negative, typically between -0.01 and -5.0 for high-confidence tokens)
        # First token gets better logprob, subsequent tokens slightly worse
        base_logprob = -0.01 - (i * 0.1)

        # Create top_logprobs with the actual token as top choice and some alternatives
        top_logprobs = [
            TopLogprob(token=token, logprob=base_logprob, bytes=token_bytes),
        ]

        # Add a couple of alternative tokens with worse logprobs
        token_stripped = token.strip()
        if len(token_stripped) > 1:
            # Alternative 1: similar token (e.g., capitalized version or slight variation)
            alt_token1 = token.replace(token_stripped[0], token_stripped[0].swapcase(), 1)
            alt_bytes1 = list(alt_token1.encode("utf-8"))
            top_logprobs.append(TopLogprob(token=alt_token1, logprob=base_logprob - 2.0, bytes=alt_bytes1))

        # Alternative 2: a different token (replace last char if possible)
        if len(token_stripped) > 1:
            alt_token2 = token.replace(token_stripped[-1], "X", 1)
        else:
            alt_token2 = token + "X"
        alt_bytes2 = list(alt_token2.encode("utf-8"))
        top_logprobs.append(TopLogprob(token=alt_token2, logprob=base_logprob - 4.0, bytes=alt_bytes2))

        token_logprob = ChatCompletionTokenLogprob(
            token=token,
            logprob=base_logprob,
            bytes=token_bytes,
            top_logprobs=top_logprobs[:3],
        )
        token_logprobs.append(token_logprob)

    return token_logprobs


@contextmanager
def patch_acompletion(mock_response: str, *, logprobs: bool = False):
    """Context manager to patch acompletion with a mocked response.

    Args:
        mock_response: The response text to include in the mock completion
        logprobs: Whether to append artificial logprobs to the mock completion

    Usage:
        with patch_acompletion("Response: [Paris]"):
            completion = await generate_completion(...)
    """

    async def mock_acompletion_with_mock_response(**kwargs):
        """Mock acompletion that adds mock_response to kwargs and calls real litellm.acompletion."""
        kwargs["mock_response"] = mock_response
        mock_completion = await litellm.acompletion(**kwargs)

        logprobs_disabled = kwargs.get("logprobs") is False
        if (
            logprobs
            and not logprobs_disabled
            and isinstance(mock_completion, ModelResponse)
            and mock_completion.choices
        ):
            token_logprobs = _generate_logprobs_from_text(mock_response)
            mock_logprobs = ChoiceLogprobs.model_construct(content=token_logprobs)
            object.__setattr__(mock_completion.choices[0], "logprobs", mock_logprobs)

        return mock_completion

    with patch("tlm.utils.completion_utils.acompletion", side_effect=mock_acompletion_with_mock_response):
        yield
