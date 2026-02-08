import json
import numpy as np
from typing import Callable

from tlm.types import FieldMetadata, Completion, SOReflectionScoreConfigType
from tlm.utils.math_utils import make_score_asymptotic
from tlm.config.presets import (
    STRUCTURED_OUTPUT_CORRECT_FIELD_SCORE,
    STRUCTURED_OUTPUT_INCORRECT_FIELD_SCORE,
)


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


def extract_incorrect_fields_reflection_metadata(
    answer: str,
    reference_answer: str,
) -> dict[str, FieldMetadata]:
    answer_json = json.loads(answer)

    incorrect_fields_list = answer_json["incorrect_fields"]
    incorrect_field_names_and_explanations = {item["field_name"]: item["explanation"] for item in incorrect_fields_list}

    field_names = json.loads(reference_answer).keys()
    per_field_metadata = {}

    # construct scores and mapped scores for each field for downstream use of per-field score details
    for field in field_names:
        if field in incorrect_field_names_and_explanations.keys():
            per_field_metadata[field] = FieldMetadata(
                score=STRUCTURED_OUTPUT_INCORRECT_FIELD_SCORE,
                explanation=incorrect_field_names_and_explanations[field],
            )
        else:
            per_field_metadata[field] = FieldMetadata(score=STRUCTURED_OUTPUT_CORRECT_FIELD_SCORE)

    return per_field_metadata


def compute_field_metadata(
    completion_metadata: list[dict[str, FieldMetadata]],
    scoring_data: list[Completion] | None = None,
) -> dict[str, FieldMetadata]:
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
        all_scores = data["scores"]
        scores_with_explanation = []
        explanations = []

        for score, explanation in zip(data["scores"], data["explanations"]):
            if explanation:
                scores_with_explanation.append(score)
                explanations.append(explanation)

        # get explanation from the SR completion with the lowest score
        if scores_with_explanation:
            min_score_idx = np.argmin(scores_with_explanation)
            explanation = explanations[min_score_idx]
        else:
            explanation = None

        # linearly rescale the scores from [min_possible_score, max_possible_score] to [0.001, 0.999]
        avg_score = float(np.mean(all_scores))

        if scoring_data is not None and len(scoring_data) > 0:
            num_total_templates = len(scoring_data)

            num_incorrect_fields_templates = sum(
                completion.template is not None
                and completion.template.so_reflection_score_config_type == SOReflectionScoreConfigType.INCORRECT_FIELDS
                for completion in scoring_data
            )

            num_non_incorrect_fields_templates = num_total_templates - num_incorrect_fields_templates

            max_possible_score = (
                STRUCTURED_OUTPUT_CORRECT_FIELD_SCORE * num_incorrect_fields_templates
                + 1.0 * num_non_incorrect_fields_templates
            ) / num_total_templates
            min_possible_score = 1 - max_possible_score

            score_range = max_possible_score - min_possible_score
            if score_range > 0:
                normalized_score = max(avg_score - min_possible_score, 0.0) / score_range
                scaled_score = make_score_asymptotic(normalized_score)
            else:
                scaled_score = avg_score
        else:
            scaled_score = avg_score

        composite_metadata[field_name] = {
            "score": scaled_score,
            "explanation": explanation,
        }

    return composite_metadata  # type: ignore[return-value]
