import pytest

from tlm.templates.reference_completion_template import ReferenceCompletionTemplate
from tlm.config.presets import ReasoningEffort


@pytest.fixture
def reference_template() -> ReferenceCompletionTemplate:
    return ReferenceCompletionTemplate.create(reasoning_effort=ReasoningEffort.NONE)


@pytest.fixture
def reference_template_with_reasoning() -> ReferenceCompletionTemplate:
    return ReferenceCompletionTemplate.create(reasoning_effort=ReasoningEffort.HIGH)
