from typing import ClassVar

from tlm.config.presets import ReasoningEffort
from tlm.templates.keywords import (
    CONTEXT_IDENTIFIER_PLACEHOLDER,
    CONTEXT_PLACEHOLDER,
    MAX_EXPLANATION_WORDS_PLACEHOLDER,
    QUERY_IDENTIFIER_PLACEHOLDER,
    QUERY_PLACEHOLDER,
    EVAL_CRITERIA_PLACEHOLDER,
    REFERENCE_ANSWER_PLACEHOLDER,
    RESPONSE_IDENTIFIER_PLACEHOLDER,
)
from tlm.templates.parsers import RATING_XML_PARSER, THINK_RATING_XML_PARSER
from tlm.templates.score_mapping import score_5_mapping
from tlm.types import Eval, CompletionTemplate


class SemanticEvaluationCompletionTemplate(CompletionTemplate):
    _PREFIX: ClassVar[str] = "## Information to consider\n\n"
    _QUERY_PROMPT: ClassVar[str] = (
        f"The {QUERY_IDENTIFIER_PLACEHOLDER} below is a user request asked to an AI Assistant.\n"
    )
    _CONTEXT_PROMPT: ClassVar[str] = (
        f"The {CONTEXT_IDENTIFIER_PLACEHOLDER} is auxiliary information retrieved to help the AI Assistant answer this user request.\n"
    )
    _RESPONSE_PROMPT: ClassVar[str] = (
        f"The {RESPONSE_IDENTIFIER_PLACEHOLDER} is the answer provided to the user by the AI Assistant.\n"
    )

    _SHARED_PROMPT: ClassVar[str] = f"""## Your Task

<criteria>
{EVAL_CRITERIA_PLACEHOLDER}
</criteria>

Evaluate the provided information based on the given criteria and rate it between 1 and 5, where 5 indicates meeting the criteria exceptionally well and 1 indicates not meeting the criteria at all.

Format your output using the following template:"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""

<think>
[Think carefully step by step to derive the rating in no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words]
</think>"""

    _SHARED_OUTPUT_FORMAT_RATING_1_5: ClassVar[str] = """

<rating>
[Single integer between 1 and 5]
</rating>"""

    _INPUT_INFORMATION_QUERY: ClassVar[str] = f"{QUERY_IDENTIFIER_PLACEHOLDER}:\n{QUERY_PLACEHOLDER}\n\n"
    _INPUT_INFORMATION_CONTEXT: ClassVar[str] = f"{CONTEXT_IDENTIFIER_PLACEHOLDER}:\n{CONTEXT_PLACEHOLDER}\n\n"
    _INPUT_INFORMATION_REFERENCE_ANSWER: ClassVar[str] = (
        f"{RESPONSE_IDENTIFIER_PLACEHOLDER}:\n{REFERENCE_ANSWER_PLACEHOLDER}\n\n"
    )

    @classmethod
    def create(cls, eval: Eval, reasoning_effort: ReasoningEffort, **kwargs) -> "SemanticEvaluationCompletionTemplate":
        prompt_parts = [cls._PREFIX]
        input_information = []

        if eval.query_identifier is not None:
            prompt_parts.append(cls._QUERY_PROMPT)
            input_information.append(cls._INPUT_INFORMATION_QUERY)

        if eval.context_identifier is not None:
            prompt_parts.append(cls._CONTEXT_PROMPT)
            input_information.append(cls._INPUT_INFORMATION_CONTEXT)

        if eval.response_identifier is not None:
            prompt_parts.append(cls._RESPONSE_PROMPT)
            input_information.append(cls._INPUT_INFORMATION_REFERENCE_ANSWER)

        if input_information:
            prompt_parts.append("\n")
            prompt_parts.extend(input_information)

        prompt_parts.append(cls._SHARED_PROMPT)

        if reasoning_effort != ReasoningEffort.NONE:
            prompt_parts.append(cls._OUTPUT_FORMAT_EXPLANATION)

            parse_patterns = THINK_RATING_XML_PARSER
        else:
            parse_patterns = RATING_XML_PARSER

        prompt_parts.append(cls._SHARED_OUTPUT_FORMAT_RATING_1_5)

        return cls(
            prompt_template="".join(prompt_parts),
            parse_patterns=parse_patterns,
            score_mapper=score_5_mapping,
            include_message_context=False,
            use_logprobs=True,
            **kwargs,
        )
