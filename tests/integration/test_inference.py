#!/usr/bin/env python3
"""
Test script for the TLM inference pipeline using PipelineFactory.create().
This script demonstrates how to create and run a complete inference pipeline.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel

# Add the project directory to Python path BEFORE importing tlm modules
tlm_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, tlm_core_path)

from tlm.config.base import ConfigInput, ReasoningEffort  # noqa: E402
from tlm.config.models import BEDROCK_MODELS  # noqa: E402
from tlm.config.presets import QualityPreset  # noqa: E402
from tlm.templates import ReferenceCompletionTemplate  # noqa: E402
from tlm.api import inference  # noqa: E402
from tlm.utils.completion_utils import generate_completion  # noqa: E402
from tlm.types import Completion, SemanticEval, SimilarityMeasure  # noqa: E402

# Load environment variables from .env file at top level of project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(project_root, ".env"))


async def run_inference_test(kwargs: dict, enabled=True) -> bool:
    """Test the inference API."""

    if not enabled:
        return True

    print("🚀 Starting TLM Inference Test")
    print(f"📝 Test completion args: {kwargs['openai_args']}")
    print("=" * 60)

    try:
        print("📝 Using configuration:")
        config_input = kwargs["config_input"]
        print(f"   - Model: {config_input.model}")
        print(f"   - Quality preset: {config_input.quality_preset}")
        print(f"   - Reasoning effort: {config_input.reasoning_effort}")
        print(f"   - Similarity measure: {config_input.similarity_measure}")
        print()

        # Run the inference
        print("🔄 Running inference...")
        print("   This may take a moment as it makes API calls...")

        response = await inference(**kwargs)

        print("✅ Inference completed!")
        print("\n📊 Response:")
        print("=" * 40)

        if isinstance(response["response"], BaseModel):
            response_dict = response["response"].model_dump()
            if "logprobs" in response_dict["choices"][0]:
                response_dict["choices"][0]["logprobs"] = "TRUNCATED"
            response_str = json.dumps(response_dict, indent=2)
        else:
            response_str = response["response"]

        print(f"   - Response: {response_str}")
        print(f"   - Confidence score: {response['confidence_score']}")
        print(f"   - Usage: {response['usage']}")
        print(f"   - Metadata: {response['metadata']}")
        print(f"   - RAG evals: {response['evals']}")
        print(f"   - Explanation: {response['explanation']}")

        print("\n🎉 Test completed successfully!")

    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


async def run_tests():
    """Run the tests."""

    test_inference_params = [
        {
            "config_input": ConfigInput(
                quality_preset=QualityPreset.BASE,
                reasoning_effort=ReasoningEffort.LOW,
                model="gpt-4.1-mini",
            ),
            "openai_args": {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
        },
        {
            "config_input": ConfigInput(
                quality_preset=QualityPreset.HIGH,
                reasoning_effort=ReasoningEffort.HIGH,
                similarity_measure=SimilarityMeasure.EMBEDDING_LARGE,
                num_reference_completions=3,
            ),
            "openai_args": {
                "messages": [{"role": "user", "content": "Explain the concept of machine learning in simple terms."}]
            },
            "evals": [
                SemanticEval(
                    name="clarity",
                    criteria="The response is clear and easy to understand.",
                    response_identifier="response",
                ),
                SemanticEval(
                    name="conciseness",
                    criteria="The response is concise and to the point.",
                    response_identifier="response",
                ),
            ],
            "enabled": True,
        },
        {
            "config_input": ConfigInput(
                quality_preset=QualityPreset.MEDIUM,
                reasoning_effort=ReasoningEffort.MEDIUM,
                similarity_measure=SimilarityMeasure.JACCARD,
            ),
            "openai_args": {
                "messages": [{"role": "user", "content": "Is this statement true or false: 'The Earth is flat.'"}]
            },
        },
        {
            "config_input": ConfigInput(
                quality_preset=QualityPreset.HIGH,
                reasoning_effort=ReasoningEffort.HIGH,
                constrain_outputs=["positive", "negative", "neutral"],
                model="gpt-4.1-mini",
            ),
            "openai_args": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Categorize the tone of this statement as positive, negative, or neutral: 'The Earth is a beautiful planet.'",
                    }
                ]
            },
        },
        {
            "config_input": ConfigInput(
                quality_preset=QualityPreset.HIGH,
                reasoning_effort=ReasoningEffort.HIGH,
                constrain_outputs=["yes", "no"],
                model="claude-3.5-sonnet",
            ),
            "openai_args": {
                "messages": [{"role": "user", "content": "Answer yes or no: Is Python a programming language?"}]
            },
        },
        {
            "config_input": ConfigInput(
                quality_preset=QualityPreset.HIGH,
                reasoning_effort=ReasoningEffort.HIGH,
            ),
            "openai_args": {
                "messages": [
                    {"role": "user", "content": "How many Rs in strawberry? Only answer with the number 2 or 4."}
                ]
            },
            # "enabled": True,
        },
    ]
    for kwargs in test_inference_params:
        enabled = kwargs.pop("enabled", False)
        await run_inference_test(kwargs, enabled=enabled)

    test_rag_params = [
        {
            "openai_args": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Answer the user Question using the Context.\n"
                            "Question: How much water does your Simple Water Bottle hold?\n"
                            "Context: The Simple Water Bottle is a reusable 27 oz water bottle."
                        ),
                    }
                ],
            },
            "context": "The Simple Water Bottle is a reusable 27 oz water bottle.",
            # "evals": DEFAULT_RAG_EVALS,
            "config_input": ConfigInput(
                quality_preset=QualityPreset.BEST,
                reasoning_effort=ReasoningEffort.MEDIUM,
            ),
        }
    ]

    for kwargs in test_rag_params:
        await run_inference_test(kwargs, enabled=False)

    test_so_scoring_params = [
        {
            "openai_args": {
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
            },
            "response": {
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
            },
            "config_input": ConfigInput(
                quality_preset=QualityPreset.HIGH,
                model="gpt-4.1-mini",
            ),
            "enabled": False,
        },
        {
            "openai_args": {
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
            },
            "config_input": ConfigInput(
                quality_preset=QualityPreset.HIGH,
                model="gpt-4.1-mini",
            ),
        },
    ]

    for kwargs in test_so_scoring_params:
        enabled = kwargs.pop("enabled", False)
        await run_inference_test(kwargs, enabled=enabled)


async def run_completion_tests_for_bedrock_models(enabled=True) -> None:
    """Run completion tests for Bedrock models."""

    if not enabled:
        return

    template = ReferenceCompletionTemplate.create(reasoning_effort=ReasoningEffort.NONE)
    for model in BEDROCK_MODELS:
        completion = await generate_completion(
            template, template_kwargs={"prompt": "What is the capital of France?"}, completion_params={"model": model}
        )
        assert isinstance(completion, Completion)
        print(f"Completion content for {model}: {completion.message}")


async def run_all_tests():
    """Run all tests in a single async context to ensure proper cleanup."""
    try:
        # Run the main tests
        success = await run_tests()
        bedrock_result = await run_completion_tests_for_bedrock_models(enabled=True)
        # run_completion_tests_for_bedrock_models returns None when disabled, treat as success
        success = success and (bedrock_result is not False)
        return success
    finally:
        # Give a small delay to allow any pending operations to complete
        await asyncio.sleep(0.1)

        # Ensure all pending tasks complete
        pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def main():
    """Main function to run the tests."""
    print("🧪 TLM Inference Pipeline Test Suite")
    print("=" * 50)

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEFAULT_API_KEY")
    if not api_key:
        print("❌ No API key found. Please set OPENAI_API_KEY or DEFAULT_API_KEY in your .env file")
        return

    print(f"✅ API key found: {api_key[:8]}...")

    # Use a single event loop for all tests to ensure proper cleanup
    success = asyncio.run(run_all_tests())

    if success:
        print("\n🏁 All tests completed!")


if __name__ == "__main__":
    main()
