from typing import ClassVar

from tlm.templates.keywords import PROMPT_PLACEHOLDER
from tlm.types import CompletionTemplate, AnswerChoiceToken
from tlm.templates.parsers import CHOICE_YES_NO_XML_PARSER
from tlm.templates.score_mapping import yes_no_mapping


class PromptAnswerabilityCompletionTemplate(CompletionTemplate):
    _PROMPT: ClassVar[
        str
    ] = f"""Your task is to measure the difficulty of user requests, not to respond to the requests.

Here is a User Request for you to evaluate:

<request>
{PROMPT_PLACEHOLDER}
</request>

The User Request may contain: instructions to follow, and retrieved context/documents that provide relevant information to help answer the request.

# Instructions

Determine: Given no additional tools, information, or context beyond what is provided in the request, will you be able to correctly answer this request? Yes or No.

Output Yes only if you are confident that you could produce a response that would be considered correct and factually accurate.

Output No if there is any chance that you might produce a response considered incorrect.

Before choosing, analyze the request carefully and consider:

1. What type of response is the request seeking? If it is seeking a non-propositional response, then choose Yes.
2. How clear and unambiguous is the request? If the wording is vague, underspecified, or potentially misleading, then choose No.
3. Does the request contain false assumptions or incorrect/nonsensical statements? If the request embeds false premises or incorrect information that would make a correct answer impossible, then choose No.
4. How tricky or challenging is the request? If the request seems difficult, requires multi-step logic, complex reasoning, careful edge-case handling, or meticulous thinking, then choose No.
5. What facts or information are needed to answer accurately, and is this information sufficiently provided in the retrieved context/documents? If the request requires information that is not adequately covered in the provided context or requires additional domain-specific, niche, or very recent information, then choose No.

Think about your limits and be self-aware. Remember that you are a hallucination-prone AI model that often gets facts wrong, misinterprets small details in user requests, and struggles with challenging questions that require many thinking steps. Carefully consider every detail of the user request and do not be over-confident. If you choose Yes, you better be right!

Your output should strictly use the following template:

<choice>
[choose one option between Yes or No]
</choice>"""

    _ANSWER_CHOICE_TOKENS: ClassVar[list[AnswerChoiceToken]] = [
        AnswerChoiceToken(token="Yes", positive=True),
        AnswerChoiceToken(token="No", positive=False),
        AnswerChoiceToken(token="yes", positive=True),
        AnswerChoiceToken(token="no", positive=False),
    ]

    @classmethod
    def create(cls, **kwargs):
        return cls(
            answer_choice_tokens=cls._ANSWER_CHOICE_TOKENS,
            prompt_template=cls._PROMPT,
            parse_patterns=CHOICE_YES_NO_XML_PARSER,
            score_mapper=yes_no_mapping,
            use_logprobs=True,
            include_message_context=False,
            **kwargs,
        )
