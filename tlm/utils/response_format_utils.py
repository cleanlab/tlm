from typing import Any, Dict
from pydantic import BaseModel, Field, create_model
import json
import copy

from tlm.types import CompletionParams
from tlm.config.defaults import get_settings

settings = get_settings()


def add_explanation_to_response_format(completion_params: CompletionParams) -> CompletionParams | None:
    if "response_format" not in completion_params:
        return None

    modified_params = copy.deepcopy(completion_params)
    json_schema = add_explanation_field(modified_params["response_format"])
    modified_params["response_format"] = json_schema

    modified_params["logprobs"] = True
    modified_params["top_logprobs"] = settings.TOP_LOGPROBS

    return modified_params


def add_explanation_field(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a top-level 'explanation' field and nests the original schema under 'answer' field."""
    original_name = schema["json_schema"]["name"]
    schema = schema["json_schema"]["schema"]
    original_defs = schema.pop("$defs", {})

    # TODO: Write up ObvConsistencyResponse class and use that to construct JSON schema?
    observed_consistency_schema = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "$defs": {**original_defs, original_name: schema},
                "properties": {
                    "explanation": {"title": "Explanation", "type": "string"},
                    "answer": {"$ref": f"#/$defs/{original_name}"},
                },
                "required": ["explanation", "answer"],
                "title": "ObvConsistencyResponse",
                "type": "object",
                "additionalProperties": False,
            },
            "name": "ObvConsistencyResponse",
            "strict": True,
        },
    }

    return observed_consistency_schema


def construct_per_field_response_format_model(
    reference_answer: str, per_field_score_response_format: type[BaseModel]
) -> type[BaseModel]:
    answer_keys = json.loads(reference_answer).keys()
    fields = {key: (per_field_score_response_format, Field(...)) for key in answer_keys}
    return create_model(per_field_score_response_format.__name__, **fields)  # type:ignore
