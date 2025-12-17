import json
import numpy as np
from typing import Callable

from tlm.types import FieldMetadata


def extract_per_field_reflection_metadata(
    answer: str,
    per_field_score_key: str,
    score_mapping: Callable[[str], float],
) -> dict[str, FieldMetadata]:
    answer_json = json.loads(answer)

    per_field_metadata = {}

    for field_name, field_data in answer_json.items():
        raw_score = field_data[per_field_score_key]
        mapped_score = score_mapping(raw_score)
        explanation = field_data["explanation"]

        per_field_metadata[field_name] = FieldMetadata(
            score=mapped_score,
            explanation=explanation,
        )

    return per_field_metadata


def compute_field_metadata(completion_metadata: list[dict[str, FieldMetadata]]) -> dict[str, FieldMetadata]:
    score_data: dict[str, dict[str, list]] = {}

    for metadata_per_field in completion_metadata:
        for field_name, metadata in metadata_per_field.items():
            if field_name not in score_data:
                score_data[field_name] = {
                    "scores": [metadata.score],
                    "explanations": [metadata.explanation],
                }
            else:
                score_data[field_name]["scores"].append(metadata.score)
                score_data[field_name]["explanations"].append(metadata.explanation)

    composite_metadata = {}
    for field_name, data in score_data.items():
        min_score_idx = np.argmin(data["scores"])
        composite_metadata[field_name] = {
            "score": np.mean(data["scores"]),
            "explanation": data["explanations"][min_score_idx],
        }

    return composite_metadata  # type: ignore[return-value]
