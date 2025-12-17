from typing import ClassVar

from tlm.config.presets import ReasoningEffort
from tlm.templates.parsers import REASONING_RESPONSE_PARSER, RESPONSE_PARSER

from tlm.templates.keywords import MAX_EXPLANATION_WORDS_PLACEHOLDER, PROMPT_PLACEHOLDER
from tlm.types import CompletionTemplate


class ReferenceCompletionTemplate(CompletionTemplate):
    _PROMPT_TEMPLATE: ClassVar[str] = (
        f"Prompt: {PROMPT_PLACEHOLDER}. Please strictly use the following template to provide answer: \n"
        "Response: [provide your response]."
    )
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        f"Prompt: {PROMPT_PLACEHOLDER}. Please strictly use the following template to provide answer: \n"
        f"Reasoning: [insert step-by-step reasoning no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words], \n"
        f"Response: [provide your response]."
    )

    @classmethod
    def create(
        cls,
        reasoning_effort: ReasoningEffort,
        constrain_outputs: list[str] | None = None,
        extract_answer: bool = False,
        **kwargs,
    ):
        if extract_answer:
            prompt_template = None
            parse_patterns = {}
        elif reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = REASONING_RESPONSE_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = RESPONSE_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            constrain_outputs=constrain_outputs,
            use_logprobs=True,
            extract_answer=extract_answer,
            **kwargs,
        )
