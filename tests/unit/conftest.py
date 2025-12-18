"""Pytest configuration and fixtures for tlm tests."""

import os
import logging
from pathlib import Path
import pytest
from dotenv import load_dotenv
import litellm
import vcr  # type: ignore[import-untyped]

from tests.helpers.completion_template_fixtures import (
    reference_template,
    reference_template_with_reasoning,
)
from tests.helpers.structured_outputs_fixtures import (
    structured_outputs_completion_params,
    structured_outputs_reference_completion,
)

# for better VCR compatibility
# litellm uses aiohttp internally, which can cause empty response bodies
# when VCR tries to record them after litellm has consumed the stream.
litellm.use_aiohttp_transport = False

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    yield


@pytest.fixture(scope="function")
async def vcr_cassette_async(request: pytest.FixtureRequest):
    """
    Properly patch async httpx for each test.

    This fixture ensures that httpx AsyncClient is patched before the test runs,
    which is critical for intercepting async HTTP requests made by litellm.
    """
    cassette_name = request.node.name + ".yaml"
    cassette_path = Path("tests/fixtures/vcr_cassettes") / cassette_name

    record_mode = "once" if not os.getenv("CI") else "none"

    my_vcr = vcr.VCR(  # type: ignore[attr-defined]
        filter_headers=["authorization"],
        match_on=["method", "scheme", "host", "port", "path"],
        decode_compressed_response=True,
    )

    with my_vcr.use_cassette(
        str(cassette_path),
        record_mode=record_mode,
    ):
        yield


# Force VCR to patch before any tests run
@pytest.fixture(scope="session", autouse=True)
def vcr_patch():
    """Ensure VCR patches httpx before LiteLLM is imported"""
    import vcr.stubs  # type: ignore[import-untyped]  # noqa: F401

    # This forces VCR to patch httpx immediately
    # Accessing httpx_stubs triggers the patching
    try:
        vcr.stubs.httpx_stubs  # type: ignore[import-untyped]  # noqa: F401
    except AttributeError:
        # httpx_stubs might not exist in all VCR versions
        # In that case, VCR should patch httpx automatically via pytest-recording
        pass
