import numpy as np
from typing import Literal

from tlm.components import Component
from tlm.types import Completion, InferenceType
from tlm.utils.math_utils import make_score_asymptotic
from tlm.utils.tokenize_utils import get_token_count
from tlm.utils.explainability_utils import get_explainability_message
from tlm.utils.completion_utils import get_cleaned_chat_completion


class ResponseAssembly(Component):
    """
    Assembles the response using context from previous components.
    This includes adding explanations, custom evals, usage, and metadata.
    """

    def __init__(
        self,
        model: str,
        response_type: Literal["answer", "completion"],
        inference_type: InferenceType,
        log_metadata: list[str] = [],
        depends_on: list[Component] | None = None,
    ):
        self.model = model
        self.response_type = response_type
        self.inference_type = inference_type
        self.log_metadata = log_metadata
        super().__init__(depends_on=depends_on)

    async def execute(self) -> None:
        trustworthiness_scores = self.execution_context.get("trustworthiness_scores")
        reference_answers = self.execution_context.get("reference_answers")
        reference_completions: list[Completion] = self.execution_context.get("reference_completions")

        best_answer_idx: int

        if np.isnan(trustworthiness_scores).all():
            best_answer_idx = 0
            average_trustworthiness_score = None
        else:
            best_answer_idx = np.nanargmax(trustworthiness_scores, axis=0)
            average_trustworthiness_score = np.nanmean(trustworthiness_scores)

        best_answer = reference_answers[best_answer_idx]
        best_completion = reference_completions[best_answer_idx]

        if average_trustworthiness_score is not None:
            make_score_asymptotic(average_trustworthiness_score)

        self.execution_context.add("best_answer_idx", best_answer_idx)

        if self.response_type == "answer":
            self.execution_context.add("best_response", best_answer)
        else:
            self.execution_context.add("best_response", get_cleaned_chat_completion(best_completion))

        self.execution_context.add("trustworthiness_score", average_trustworthiness_score)

        if self.inference_type == InferenceType.PROMPT:
            if best_completion.usage is None:
                prompt = self.execution_context.get("prompt", "")
                prompt_tokens = get_token_count(prompt, self.model)
                completion_tokens = get_token_count(best_answer, self.model)
            else:
                prompt_tokens = best_completion.usage.prompt_tokens
                completion_tokens = best_completion.usage.completion_tokens
            self.execution_context.add(
                "usage",
                {
                    "num_input_tokens": prompt_tokens,
                    "num_output_tokens": completion_tokens,
                },
            )

        self_reflection_completions = self.execution_context.get("self_reflection_completions")
        consistency_scores = self.execution_context.get("consistency_scores")
        observed_consistency_completions = self.execution_context.get("consistency_completions")
        consistency_scores_flat = self.execution_context.get("consistency_scores_flat")
        num_reference_answers = len(reference_answers)
        if consistency_scores_flat.size > 0:
            consistency_scores_for_best_answer = consistency_scores_flat.reshape(num_reference_answers, -1)[
                best_answer_idx
            ]
        else:
            consistency_scores_for_best_answer = np.array([])

        if np.isnan(consistency_scores).all():
            mean_consistency_score = np.nan
        else:
            mean_consistency_score = float(np.nanmean(consistency_scores))

        explainability_message = get_explainability_message(
            average_trustworthiness_score,
            self_reflection_completions,
            observed_consistency_completions,
            mean_consistency_score,
            consistency_scores_for_best_answer,
            best_answer_idx,
            best_answer,
        )
        self.execution_context.add("explanation", explainability_message)
