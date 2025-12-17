from typing import ClassVar

from tlm.config.presets import ReasoningEffort
from tlm.templates.keywords import MAX_EXPLANATION_WORDS_PLACEHOLDER, QUESTION_PLACEHOLDER
from tlm.types import CompletionTemplate
from tlm.templates.parsers import ANSWER_XML_PARSER, THINK_ANSWER_XML_PARSER


class ObservedConsistencyQACompletionTemplate(CompletionTemplate):
    _PROMPT_TEMPLATE: ClassVar[str] = f"{QUESTION_PLACEHOLDER}"
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = f"""{QUESTION_PLACEHOLDER}


## Additional instructions for formatting your response

Please strictly use the following template to format your response:

<think>
[think carefully step by step (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words) before providing your answer]
</think>

<answer>
[provide your final answer]
</answer>"""

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
            parse_patterns = THINK_ANSWER_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = ANSWER_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            constrain_outputs=constrain_outputs,
            use_logprobs=constrain_outputs is not None,
            extract_answer=extract_answer,
            **kwargs,
        )
