from tlm.components import Component
from tlm.utils.scoring.self_reflection_scoring_utils import generate_self_reflection_scores
from tlm.types import Completion
from tlm.utils.scoring.per_field_scoring_utils import compute_field_metadata


class SelfReflectionScoreComputation(Component):
    async def execute(self) -> None:
        reference_answers: list[str] = self.execution_context.get("reference_answers")
        self_reflection_completions: list[list[Completion]] = self.execution_context.get("self_reflection_completions")

        self_reflection_completions_flat = [
            completion for sublist in self_reflection_completions for completion in sublist
        ]
        self_reflection_scores = generate_self_reflection_scores(reference_answers, self_reflection_completions_flat)

        self.execution_context.add("self_reflection_scores", self_reflection_scores)

        reflection_metadata = [
            metadata
            for completion in self_reflection_completions_flat
            if (metadata := completion.per_field_metadata) is not None
        ]
        composite_reflection_metadata = compute_field_metadata(reflection_metadata)

        self.execution_context.add("self_reflection_metadata_per_field", composite_reflection_metadata)
