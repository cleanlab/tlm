import contextlib
from typing import AsyncGenerator, List, Literal, Dict, Any, cast
import logging
import httpx
import ast
import json
from openai import AsyncOpenAI

DEFAULT_COMPLETION_RETRY_ATTEMPTS = 2
DEFAULT_EMBEDDING_TIMEOUT = 5.0

CHAT_COMPLETION: Literal["chat_completion"] = "chat_completion"
PERPLEXITY: Literal["perplexity"] = "perplexity"


logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def get_openai_client() -> AsyncGenerator[AsyncOpenAI, None]:
    """Constructs and returns async OpenAI client."""
    async with httpx.AsyncClient() as http_client:
        async with AsyncOpenAI(
            # api_key=OPENAI_API_KEY,
            http_client=http_client,
            max_retries=DEFAULT_COMPLETION_RETRY_ATTEMPTS,
        ) as openai_client:
            yield openai_client


async def get_text_embedding(
    openai_client: AsyncOpenAI,
    text: str,
    model: str,
) -> List[float]:
    embedding = await openai_client.embeddings.create(
        input=text,
        model=model,
        timeout=DEFAULT_EMBEDDING_TIMEOUT,
    )
    return embedding.data[0].embedding


def extract_message_content(completion: Dict[str, Any]) -> str:
    return cast(str, completion[CHAT_COMPLETION]["choices"][0]["message"]["content"])


def extract_structured_output_field(message_content: str, field: str) -> str | None:
    try:
        return str(ast.literal_eval(message_content)[field])
    except Exception as e:
        logger.warning(f"ast.literal_eval failed for message_content: {message_content}\nError: {e}")

    try:
        return str(json.loads(message_content)[field])
    except Exception as e:
        logger.warning(f"json.loads failed for message_content: {message_content}\nError: {e}")

    return None
