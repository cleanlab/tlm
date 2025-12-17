import asyncio
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from tlm.templates.llm_consistency_completion_templates import LLMConsistencyCompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.utils.errors import LLMConsistencyInferenceError
from tlm.types import Completion, ExtractedResponseField


async def get_llm_consistency_scores(
    reference_answers: Sequence[str],
    comparison_answers: Sequence[str],
    completion_template: LLMConsistencyCompletionTemplate,
) -> npt.NDArray[np.float64]:
    try:
        completion_tasks = [
            asyncio.create_task(
                generate_completion(completion_template, template_kwargs={"input_1": reference, "input_2": comparison})
            )
            for reference, comparison in zip(reference_answers, comparison_answers)
        ]

        completions = await asyncio.gather(*completion_tasks)
    except Exception as e:
        raise LLMConsistencyInferenceError(f"Failed to get LLM consistency scores: {e}")

    return np.array(
        [
            completion.response_fields.get(ExtractedResponseField.MAPPED_SCORE, np.nan)
            if isinstance(completion, Completion)
            else np.nan
            for completion in completions
        ]
    )
