import asyncio

from tlm.components import Component
from tlm.config.presets import REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS, ReasoningEffort
from tlm.templates import SemanticEvaluationCompletionTemplate
from tlm.utils.completion_utils import generate_completion
from tlm.utils.scoring.semantic_evaluation_scoring_utils import compute_semantic_evaluation_scores
from tlm.types import Eval


class SemanticEvaluationScoreGenerator(Component):
    """
    Generates completions and computes evaluation scores using LLM-as-judge with semantic criteria.
    """

    def __init__(
        self,
        query: str | None,
        context: str | None,
        evals: list[Eval],
        reasoning_effort: ReasoningEffort,
        temperature: float,
        **kwargs,
    ):
        query_required = any(eval.query_identifier is not None for eval in evals)
        if query_required and query is None:
            raise ValueError("query must be provided if any evals require it")

        context_required = any(eval.context_identifier is not None for eval in evals)
        if context_required and context is None:
            raise ValueError("context must be provided if any evals require it")

        self.query = query
        self.context = context
        self.evals = evals
        self.reasoning_effort = reasoning_effort
        self.max_explanation_words = REASONING_EFFORT_TO_MAX_EXPLANATION_WORDS[reasoning_effort]
        self.temperature = temperature
        super().__init__(**kwargs)

    async def execute(self) -> None:
        if not self.evals:
            return

        use_reference_answers = any(eval.response_identifier is not None for eval in self.evals)

        reference_answers: list[str | None] = (
            self.execution_context.get("reference_answers") if use_reference_answers else [None]
        )

        params = []
        for answer in reference_answers:
            for eval in self.evals:
                params.append((eval, answer))

        completion_tasks = [
            asyncio.create_task(
                generate_completion(
                    template=SemanticEvaluationCompletionTemplate.create(
                        eval=eval, reasoning_effort=self.reasoning_effort
                    ),
                    template_kwargs={
                        "query_identifier": eval.query_identifier,
                        "context_identifier": eval.context_identifier,
                        "response_identifier": eval.response_identifier,
                        "query": self.query,
                        "context": self.context,
                        "reference_answer": reference_answer,
                        "eval_criteria": eval.criteria,
                        "max_explanation_words": self.max_explanation_words,
                    },
                    temperature=self.temperature,
                )
            )
            for eval, reference_answer in params
        ]

        semantic_evaluation_completions = await asyncio.gather(*completion_tasks)
        computed_scores = compute_semantic_evaluation_scores(
            reference_answers,
            self.evals,
            semantic_evaluation_completions,
        )

        context_key = "evals_requiring_response" if use_reference_answers else "evals_not_requiring_response"
        self.execution_context.add(context_key, computed_scores)
