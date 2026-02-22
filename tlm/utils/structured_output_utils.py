import ast
import json

from tlm.inference import InferenceResult


def _get_untrustworthy_fields(
    tlm_result: InferenceResult,
    threshold: float = 0.8,
    display_details: bool = True,
) -> list[str]:
    tlm_metadata = tlm_result["metadata"]
    response_text = tlm_result["response"].choices[0].message.content  # type: ignore

    if tlm_metadata is None or "per_field_score" not in tlm_metadata:
        raise ValueError(
            "`per_field_score` is not present in the metadata.\n"
            "`get_untrustworthy_fields()` can only be called scoring structured outputs responses."
        )

    try:
        so_response = json.loads(response_text)
    except Exception:
        pass
    try:
        so_response = ast.literal_eval(response_text)
    except Exception:
        raise ValueError(
            "The LLM response must be a valid JSON output (use `response_format` to specify the output format)"
        )

    per_field_score = tlm_metadata["per_field_score"]
    per_score_details = []

    # handle cases where error log is returned
    if len(per_field_score) == 1 and isinstance(per_field_score.get("error"), str):
        print("Per-field score returned an error:")
        print(per_field_score.get("error"))
        return []

    for key, value in per_field_score.items():
        score = value["score"]
        if float(score) < threshold:
            key_details = {
                "response": so_response[key],
                "score": score,
                "explanation": value["explanation"],
            }
            per_score_details.append({key: key_details})

    per_score_details.sort(key=lambda x: next(iter(x.values()))["score"])
    untrustworthy_fields = [next(iter(item.keys())) for item in per_score_details]

    if display_details:
        if len(untrustworthy_fields) == 0:
            print("No untrustworthy fields found")

        else:
            print(f"Untrustworthy fields: {untrustworthy_fields}\n")
            for item in per_score_details:
                print(f"Field: {next(iter(item.keys()))}")
                details = next(iter(item.values()))
                print(f"Response: {details['response']}")
                print(f"Score: {details['score']}")
                print(f"Explanation: {details['explanation']}")
                print()

    return untrustworthy_fields
