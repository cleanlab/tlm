from typing import Any, Dict

from tlm.components import (
    TrustworthinessScoreComputation,
    ConsistencyScoreComputation,
    ObservedConsistencyCompletionGenerator,
    PerplexityScoreComputation,
    PromptEvaluationCompletionGenerator,
    PromptEvaluationScoreExtraction,
    SemanticEvaluationScoreGenerator,
    ReferenceCompletionFormatter,
    ReferenceCompletionGenerator,
    ResponseAssembly,
    SelfReflectionCompletionGenerator,
    SelfReflectionScoreComputation,
)
from tlm.config.base import BaseConfig
from tlm.config.presets import WorkflowType
from tlm.pipeline import InferencePipeline
from tlm.utils.prompt_utils import format_user_request, extract_user_prompt
from tlm.utils.eval_utils import group_evals
from tlm.types import Eval, CompletionParams, InferenceType


class PipelineFactory:
    @staticmethod
    def create(
        *,
        completion_params: CompletionParams,
        config: BaseConfig,
        response: Dict[str, Any] | None,
        evals: list[Eval] | None,
        context: str | None,
    ) -> InferencePipeline:
        pipeline = InferencePipeline()

        user_prompt = extract_user_prompt(completion_params)
        user_request = format_user_request(completion_params)

        if config.use_prompt_evaluation:
            prompt_evaluation_completion_generator = pipeline.add(
                PromptEvaluationCompletionGenerator(
                    prompt=user_prompt,
                    temperature=config.prompt_evaluation_temperature,
                )
            )
        else:
            prompt_evaluation_completion_generator = None

        observed_consistency_completion_generator = pipeline.add(
            ObservedConsistencyCompletionGenerator(
                completion_params=completion_params,
                count=config.num_consistency_completions,
                temperature=config.observed_consistency_temperature,
                reasoning_effort=config.reasoning_effort,
                constrain_outputs=config.constrain_outputs,
            )
        )

        evals_requiring_response, evals_not_requiring_response = group_evals(evals)

        evals_not_requiring_response_generator = (
            pipeline.add(
                SemanticEvaluationScoreGenerator(
                    query=user_prompt,
                    context=context,
                    evals=evals_not_requiring_response,
                    reasoning_effort=config.reasoning_effort,
                    temperature=config.semantic_evaluation_temperature,
                    model=config.model,
                )
            )
            if evals_not_requiring_response
            else None
        )

        inference_type = InferenceType.SCORE if response else InferenceType.PROMPT

        reference_completion_component = pipeline.add(
            ReferenceCompletionFormatter(completion_params=completion_params, response_input=response)
            if inference_type == InferenceType.SCORE and response
            else ReferenceCompletionGenerator(
                count=config.num_reference_completions,
                min_count=config.min_reference_completions,
                completion_params=completion_params,
                reasoning_effort=config.reasoning_effort,
                constrain_outputs=config.constrain_outputs,
            )
        )

        self_reflection_completion_generator = pipeline.add(
            SelfReflectionCompletionGenerator(
                prompt=user_request,
                reasoning_effort=config.reasoning_effort,
                workflow_type=config.workflow_type,
                num_completions=config.num_self_reflection_completions,
                depends_on=[reference_completion_component],
            )
        )

        evals_requiring_response_generator = (
            pipeline.add(
                SemanticEvaluationScoreGenerator(
                    query=user_prompt,
                    context=context,
                    evals=evals_requiring_response,
                    reasoning_effort=config.reasoning_effort,
                    temperature=config.semantic_evaluation_temperature,
                    model=config.model,
                    depends_on=[reference_completion_component],
                )
            )
            if evals_requiring_response
            else None
        )

        consistency_score_computation = pipeline.add(
            ConsistencyScoreComputation(
                similarity_measure=config.similarity_measure,
                structured_outputs=(config.workflow_type == WorkflowType.STRUCTURED_OUTPUT_SCORING),
                constrain_outputs=config.constrain_outputs,
                depends_on=[reference_completion_component, observed_consistency_completion_generator],
            )
        )
        perplexity_score_computation = pipeline.add(
            PerplexityScoreComputation(depends_on=[reference_completion_component])
        )
        self_reflection_score_computation = pipeline.add(
            SelfReflectionScoreComputation(
                depends_on=[reference_completion_component, self_reflection_completion_generator]
            )
        )

        if prompt_evaluation_completion_generator:
            prompt_evaluation_score_extraction = pipeline.add(
                PromptEvaluationScoreExtraction(
                    depends_on=[prompt_evaluation_completion_generator, reference_completion_component],
                )
            )
        else:
            prompt_evaluation_score_extraction = None

        trustworthiness_score_computation = pipeline.add(
            TrustworthinessScoreComputation(
                workflow_type=config.workflow_type,
                model=config.model,
                depends_on=[
                    component
                    for component in [
                        consistency_score_computation,
                        perplexity_score_computation,
                        self_reflection_score_computation,
                        prompt_evaluation_score_extraction,
                    ]
                    if component is not None
                ],
            )
        )

        pipeline.add(
            ResponseAssembly(
                model=config.model,
                response_type="completion",
                inference_type=inference_type,
                depends_on=[
                    component
                    for component in [
                        trustworthiness_score_computation,
                        evals_not_requiring_response_generator,
                        evals_requiring_response_generator,
                    ]
                    if component is not None
                ],
            )
        )

        return pipeline
