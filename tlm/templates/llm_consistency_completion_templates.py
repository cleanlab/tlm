from typing import ClassVar

from tlm.templates.keywords import INPUT_1_PLACEHOLDER, INPUT_2_PLACEHOLDER
from tlm.types import CompletionTemplate
from tlm.templates.parsers import ANSWER_YES_NO_XML_PARSER, CHOICE_A_B_XML_PARSER

from tlm.templates.score_mapping import ab_mapping, yes_no_mapping


class LLMConsistencyCompletionTemplate(CompletionTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CodeConsistencyCompletionTemplate(LLMConsistencyCompletionTemplate):
    _PROMPT_TEMPLATE: ClassVar[
        str
    ] = f"""You are an expert code reviewer. Please analyze these two code snippets and determine if they are functionally equivalent (i.e. they accomplish the same task and would produce the same outputs given the same inputs).

Snippet 1:
{INPUT_1_PLACEHOLDER}

Snippet 2:
{INPUT_2_PLACEHOLDER}

Are these code snippets functionally equivalent? Answer with A for Yes and B for No.

The output should strictly use the following template:
Choice: [choose one letter between A and B]"""

    @classmethod
    def create(cls):
        return cls(
            prompt_template=cls._PROMPT_TEMPLATE,
            parse_patterns=CHOICE_A_B_XML_PARSER,
            score_mapper=ab_mapping,
            use_logprobs=False,
            include_message_context=False,
        )


class StatementConsistencyCompletionTemplate(LLMConsistencyCompletionTemplate):
    _PROMPT_TEMPLATE: ClassVar[
        str
    ] = f"""Determine whether the two statements below, which are answers to the same prompt, are consistent with each other.

<statement_1>
{INPUT_1_PLACEHOLDER}
</statement_1>

<statement_2>
{INPUT_2_PLACEHOLDER}
</statement_2>


## Question

Are these statements consistent with each other? Answer with Yes or No.

The statements are consistent if:
- They provide identical information with zero contradictions
- They express the same idea or answer the prompt in a similar way, referring to the same fact or claim
- For short phrases or single words, they must be exact matches or clear synonyms.

If the statements describe different things or seem like they are responding to different prompts, they are not consistent even if they don't contradict each other.

The output should strictly use the following template:
<answer>
[choose Yes or No]
</answer>"""

    @classmethod
    def create(cls):
        return cls(
            prompt_template=cls._PROMPT_TEMPLATE,
            parse_patterns=ANSWER_YES_NO_XML_PARSER,
            score_mapper=yes_no_mapping,
            use_logprobs=False,
        )
