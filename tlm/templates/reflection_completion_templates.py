from abc import ABC, abstractmethod
from typing import Callable, ClassVar

from pydantic import BaseModel

from tlm.config.presets import ReasoningEffort, WorkflowType
from tlm.templates.keywords import (
    ANSWER_PLACEHOLDER,
    MAX_EXPLANATION_WORDS_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
)
from tlm.templates.parsers import (
    CHOICE_TRUE_FALSE_XML_PARSER,
    RATING_10_XML_PARSER,
    RATING_XML_PARSER,
    SCORE_XML_PARSER,
    SCORE_10_XML_PARSER,
    THINK_CHOICE_XML_PARSER,
    THINK_RATING_10_XML_PARSER,
    THINK_RATING_XML_PARSER,
    THINK_SCORE_XML_PARSER,
    ISSUES_SCORE_XML_PARSER,
    CHOICE_AB_PARSER,
    RATING_1_5_PARSER,
)
from tlm.templates.score_mapping import (
    score_5_mapping,
    score_10_mapping,
    score_100_mapping,
    true_false_mapping,
    ab_mapping,
    certainty_mapping,
)
from tlm.types import AnswerChoiceToken, ExtractedResponseField, RegexPattern, CompletionTemplate
from tlm.utils.response_format_utils import construct_per_field_response_format_model
from tlm.templates.per_field_scoring_models import PerFieldCorrectnessEvaluation, PerFieldCertaintyEvaluation


class ReflectionCompletionTemplate(CompletionTemplate, ABC):
    _SHARED_USER_REQUEST_PROPOSED_RESPONSE_PROMPT: ClassVar[str] = (
        "Below is a User Request and the Response proposed by an unreliable AI Assistant:\n\n"
        f"<request>\n{QUESTION_PLACEHOLDER}\n</request>\n\n"
        f"<response>\n{ANSWER_PLACEHOLDER}\n</response>\n\n\n"
    )

    def __init__(
        self,
        prompt_template: str,
        parse_patterns: dict[ExtractedResponseField, list[RegexPattern]],
        answer_choice_tokens: list[AnswerChoiceToken] = [],
        score_mapper: Callable[[str], float] | None = None,
        temperature: float | None = None,
        use_logprobs: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            answer_choice_tokens=answer_choice_tokens,
            score_mapper=score_mapper,
            temperature=temperature,
            use_logprobs=use_logprobs,
            include_message_context=False,
            **kwargs,
        )

    @classmethod
    @abstractmethod
    def create(
        cls,
        reasoning_effort: ReasoningEffort,
        **kwargs,
    ) -> "ReflectionCompletionTemplate": ...


class ReflectionSOPerFieldScoreTemplate(ReflectionCompletionTemplate):
    per_field_score_response_format: ClassVar[type[BaseModel]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "per_field_score_response_format"):
            raise TypeError(f"{cls.__name__} must define 'per_field_score_response_format' class variable")

    @classmethod
    def construct_response_format(cls, response_json: str) -> type[BaseModel] | None:
        return construct_per_field_response_format_model(response_json, cls.per_field_score_response_format)


class ReflectionCertaintyTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = f"""You are an evaluator that verifies whether AI responses are factually correct.
Below is a User Question and the Answer proposed by an untrustworthy AI Assistant.
Your task is to check whether the proposed Answer is correct.

<question>
{QUESTION_PLACEHOLDER}
</question>

<answer>
{ANSWER_PLACEHOLDER}
</answer>


# Instructions

How certain are you that the proposed Answer is correct?
Provide a score between 0-100 indicating your estimated probability that the Answer is accurate.
If the proposed Answer is obviously incorrect, then rate a score of 0.

Your output should strictly use the following template:\n\n"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[think carefully step by step (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words), considering factors like: the provenance of any information/facts in the Answer, reasons why the Answer could be incorrect, alternative better Answers, etc.]
</think>"""

    _SHARED_OUTPUT_FORMAT_SCORE_0_100: ClassVar[str] = """<score>
[choose a score between 0-100]
</score>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_SCORE_0_100
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_SCORE_0_100
    )
    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_SCORE_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = SCORE_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_100_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionKnowledgeGapTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = (
        ReflectionCompletionTemplate._SHARED_USER_REQUEST_PROPOSED_RESPONSE_PROMPT
        + """How can I be sure this response is correct?

Please give a convincing argument that includes the evidence supporting this response, how you know you can trust that evidence, how you know we can trust your knowledge.

Now rate your own confidence on a scale of 0-10 that the response is correct.

Your output should strictly follow this template:\n\n"""
    )

    _SHARED_OUTPUT_FORMAT_RATING_0_10: ClassVar[str] = """<rating>
[single integer between 0 and 10]
</rating>"""

    _OUTPUT_FORMAT_RATING_EXPLANATION: ClassVar[str] = f"""<think>
[think carefully step by step to derive the rating in no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words]
</think>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_RATING_0_10
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_RATING_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_RATING_0_10
    )
    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_RATING_10_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = RATING_10_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_10_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionArgumentTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[
        str
    ] = f"""Below is a User Request and the proposed Answer from an untrustworthy AI assistant:

<request>
{QUESTION_PLACEHOLDER}
</request>

<answer>
{ANSWER_PLACEHOLDER}
</answer>


# Instructions

Think critically to provide a comprehensive veritable argument for why the proposed Answer might be incorrect.

Then consider the argument, consider the Answer again, and determine: How certain are you that the proposed Answer is correct?

Provide a score between 0-100 indicating your confidence that the Answer is correct and factually accurate.
If the proposed Answer is obviously incorrect, then rate a score of 0.

Your output should strictly use the following template:\n\n"""

    _SHARED_OUTPUT_FORMAT_SCORE_0_100: ClassVar[str] = """<score>
[choose a confidence score between 0-100]
</score>"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[argue why the proposed Answer might be incorrect (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words)]
</think>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_SCORE_0_100
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_SCORE_0_100
    )
    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_SCORE_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = SCORE_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_100_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionBinaryCorrectnessTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[
        str
    ] = f"""Below is an AI Assistant's proposed Response to a given Request. Your task is to check whether the proposed Response is correct.

<request>
{QUESTION_PLACEHOLDER}
</request>

<response>
{ANSWER_PLACEHOLDER}
</response>


Is the proposed Response correct? Answer True or False.
This AI Assistant often gives convincing but inaccurate Responses. Verify all claims rigorously, only choose True if you are confident they are correct.
If the Response is non-propositional, choose True.

Your output should strictly use the following template:\n\n"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[think carefully step by step (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words), considering factors like: the provenance of any information/facts in the response, reasons why the response could be incorrect, alternative better responses, etc.]
</think>"""

    _SHARED_OUTPUT_FORMAT_CHOICE: ClassVar[str] = """<choice>
[choose True or False]
</choice>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_CHOICE
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_CHOICE
    )
    _ANSWER_CHOICE_TOKENS: ClassVar[list[AnswerChoiceToken]] = [
        AnswerChoiceToken(token="True", positive=True),
        AnswerChoiceToken(token="False", positive=False),
        AnswerChoiceToken(token="true", positive=True),
        AnswerChoiceToken(token="false", positive=False),
    ]

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_CHOICE_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = CHOICE_TRUE_FALSE_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            answer_choice_tokens=cls._ANSWER_CHOICE_TOKENS,
            score_mapper=true_false_mapping,
            use_logprobs=True,
            **kwargs,
        )


class ReflectionTrustworthinessTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[
        str
    ] = f"""Below is an AI Assistant's proposed Response to a given Request. Your task is to check whether the proposed Response is correct.

<request>
{QUESTION_PLACEHOLDER}
</request>

<response>
{ANSWER_PLACEHOLDER}
</response>


Is the proposed Response correct? Answer True or False.
This AI Assistant often gives convincing but inaccurate Responses. Verify all claims rigorously, only choose True if you are confident they are correct.
If the Response is non-propositional, choose True.

Your output should strictly use the following template:\n\n"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[think carefully step by step (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words), considering questions such as: How can you verify this is correct? If the Response provides information, what is the source of this information?]
</think>"""

    _SHARED_OUTPUT_FORMAT_RATING_1_5: ClassVar[str] = """<rating>
[rating between 1 and 5]
</rating>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_RATING_1_5
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_RATING_1_5
    )
    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_RATING_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = RATING_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_5_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionCorrectnessTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = f"""You are a verifier of AI responses that are potentially wrong.
Below is a User Request and the Response proposed by an AI Assistant:

<request>
{QUESTION_PLACEHOLDER}
</request>

<response>
{ANSWER_PLACEHOLDER}
</response>


# Instructions

Carefully verify the accuracy of each claim in the proposed Response.
Also scrutinize the Request for tricky aspects and think about why the Response could be considered wrong.

Finally, rate the correctness of the AI Response on a scale of 0 to 10.

Your output must strictly use the following template:\n\n"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[think carefully step by step (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words), considering questions such as: How can you verify this is correct?]
</think>"""

    _SHARED_OUTPUT_FORMAT_RATING_0_10: ClassVar[str] = """<rating>
[single integer between 0 and 10]
</rating>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_RATING_0_10
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_RATING_0_10
    )
    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_RATING_10_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = RATING_10_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_10_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionRAGCertaintyTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = (
        ReflectionCompletionTemplate._SHARED_USER_REQUEST_PROPOSED_RESPONSE_PROMPT
        + """# Instructions

How certain are you that the proposed Response is correct?
Provide a score between 0-100 indicating your estimated probability that the Response is accurate.
If the proposed Response is obviously incorrect, then rate a score of 0.

Your output should strictly follow this template:

"""
    )

    _OUTPUT_FORMAT_SCORE_0_100: ClassVar[str] = """<score>
[choose a score between 0-100]
</score>"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[list reasons the response could be incorrect, and an alternate plausible response, all within {MAX_EXPLANATION_WORDS_PLACEHOLDER} words]
</think>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _OUTPUT_FORMAT_SCORE_0_100
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _OUTPUT_FORMAT_SCORE_0_100
    )

    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_SCORE_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = SCORE_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_100_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionRAGArgumentTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = (
        ReflectionCompletionTemplate._SHARED_USER_REQUEST_PROPOSED_RESPONSE_PROMPT
        + """# Instructions

Consider the following:
1. Identify any comprehensive arguments that could show why the proposed Response might be incorrect
2. Look for evidence that contradicts or challenges the Response's accuracy

Based on your analysis above:
Consider the argument and the Response again - how certain are you that the proposed Response is correct?
Provide a score between 0-10 indicating your confidence that the Response is correct and factually accurate.
If the proposed Response is obviously incorrect, then rate a score of 0.

Your output should strictly use the following template:

"""
    )

    _SHARED_OUTPUT_FORMAT_SCORE_0_10: ClassVar[str] = """<score>
[choose a confidence score between 0-10]
</score>"""

    _OUTPUT_FORMAT_EXPLANATION: ClassVar[str] = f"""<think>
[argue why the proposed Response might be incorrect (no more than {MAX_EXPLANATION_WORDS_PLACEHOLDER} words)]
</think>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_SCORE_0_10
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_EXPLANATION + "\n\n" + _SHARED_OUTPUT_FORMAT_SCORE_0_10
    )

    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = THINK_SCORE_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = SCORE_10_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_10_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionRAGIssuesTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = f"""User's original request
<request>
{QUESTION_PLACEHOLDER}
</request>

Unreliable AI Assistant's proposed answer
<response>
{ANSWER_PLACEHOLDER}
</response>

# Instructions

Consider the following:
1. What potential issues could make this response incorrect?
2. What would be an alternate plausible answer?

Based on your analysis above:
Provide a score between 0-100 indicating your estimated probability that the Response is accurate.
If the proposed Response is obviously incorrect, then rate a score of 0.

Your output should strictly follow this template:

"""

    _OUTPUT_FORMAT_ISSUES: ClassVar[str] = """<issues>
- [possible issue 1]
- [possible issue 2]
</issues>

<alternate_response>
[1-3 sentence alternate answer]
</alternate_response>

"""

    _SHARED_OUTPUT_FORMAT_SCORE_0_100: ClassVar[str] = """<score>
[choose a score between 0-100]
</score>"""

    _PROMPT_TEMPLATE: ClassVar[str] = _SHARED_PROMPT + _SHARED_OUTPUT_FORMAT_SCORE_0_100
    _PROMPT_TEMPLATE_WITH_REASONING: ClassVar[str] = (
        _SHARED_PROMPT + _OUTPUT_FORMAT_ISSUES + _SHARED_OUTPUT_FORMAT_SCORE_0_100
    )

    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort != ReasoningEffort.NONE:
            prompt_template = cls._PROMPT_TEMPLATE_WITH_REASONING
            parse_patterns = ISSUES_SCORE_XML_PARSER
        else:
            prompt_template = cls._PROMPT_TEMPLATE
            parse_patterns = SCORE_XML_PARSER

        return cls(
            prompt_template=prompt_template,
            parse_patterns=parse_patterns,
            score_mapper=score_100_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionClassificationCorrectnessTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = f"""Here is the answer an AI assistant generated for the following request:

Request: {QUESTION_PLACEHOLDER}

Answer: {ANSWER_PLACEHOLDER}

Consider reasons why the predicted Answer could be wrong and consider alternative possible answers.
Then determine how confident you are that the original predicted Answer is correct, by making a choice between:
(A) highly confident, (B) not confident.
Choose (B) if you believe the original predicted Answer may be wrong.

Respond in the following format:
Choice: [a single choice between A or B]
"""

    _ANSWER_CHOICE_TOKENS: ClassVar[list[AnswerChoiceToken]] = [
        AnswerChoiceToken(token="A", positive=True),
        AnswerChoiceToken(token="B", positive=False),
        AnswerChoiceToken(token="a", positive=True),
        AnswerChoiceToken(token="b", positive=False),
    ]

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        return cls(
            prompt_template=cls._SHARED_PROMPT,
            parse_patterns=CHOICE_AB_PARSER,
            score_mapper=ab_mapping,
            answer_choice_tokens=cls._ANSWER_CHOICE_TOKENS,
            use_logprobs=True,
        )


class ReflectionClassificationScoringTemplate(ReflectionCompletionTemplate):
    _SHARED_PROMPT: ClassVar[str] = f"""Here is the answer an AI assistant generated for the following request:

Request: {QUESTION_PLACEHOLDER}

Answer: {ANSWER_PLACEHOLDER}

Consider reasons why the predicted Answer could be wrong and consider alternative possible answers.

Then rate your confidence in the original predicted Answer on a scale of 1-5. 1 indicates you are not at all confident in the original predicted Answer, 5 indicates you are 100% confident in the original predicted Answer, with ratings 2-4 indicating increasing levels of confidence.

Respond in the following format:
Rating: [a single value between 1 to 5]"""

    _TEMPERATURE: ClassVar[float] = 0.0

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        return cls(
            prompt_template=cls._SHARED_PROMPT,
            parse_patterns=RATING_1_5_PARSER,
            score_mapper=score_5_mapping,
            temperature=cls._TEMPERATURE,
            use_logprobs=False,
            **kwargs,
        )


class ReflectionSOPerScoreCorrectnessTemplate(ReflectionSOPerFieldScoreTemplate):
    _PROMPT: ClassVar[str] = f"""Below is a User Request and the proposed Response from an untrustworthy AI assistant.
Your task is to double-check the accuracy of each field in the Response JSON.

<request>
{QUESTION_PLACEHOLDER}
</request>

<response>
{ANSWER_PLACEHOLDER}
</response>


## Instructions

For each top-level key in the JSON:
Consider whether its value in the proposed JSON Response above seems: potentially incorrect, hard for you to verify based on available information, or missing when it shouldn't be.
Then decide how confident you are that its value is actually correct, choosing one confidence rating out of these choices:
- Certain
- Mostly Certain
- Somewhat Certain
- Uncertain
- Likely Incorrect


## Output Format

Output a JSON object with the same top-level keys as the proposed JSON Response.
Each top-level key should map to an object with two fields:
- `explanation`: Briefly explain why the value in the proposed JSON Response could be incorrect, incomplete, difficult to verify, or why you are certain it is right
- `confidence`: One choice out of: [Certain, Mostly Certain, Somewhat Certain, Uncertain, Likely Incorrect]"""

    per_field_score_response_format: ClassVar[type[BaseModel]] = PerFieldCorrectnessEvaluation

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort == ReasoningEffort.NONE:
            raise ValueError("Per-field scoring only supports reasoning")

        return cls(
            prompt_template=cls._PROMPT,
            parse_patterns={},
            score_mapper=certainty_mapping,
            use_logprobs=False,
            per_field_score_key="confidence",
            **kwargs,
        )


class ReflectionSOPerScoreCertaintyTemplate(ReflectionSOPerFieldScoreTemplate):
    _PROMPT: ClassVar[str] = f"""You are a trustworthy verifier of AI-generated JSON responses.
Below is a User Request and the proposed JSON Response from an untrustworthy AI assistant.
Your task is to double check whether each field in the proposed JSON Response is correct.

<request>
{QUESTION_PLACEHOLDER}
</request>

<response>
{ANSWER_PLACEHOLDER}
</response>


## Instructions

Help me determine how much I can trust the value of each field in the proposed JSON Response.
It is crucial that you help me catch any inaccurate values.

For each top-level key in the proposed JSON Response:
- How certain are you that its value in the proposed JSON Response is entirely correct?
- Provide a score between 0-100 for this key, indicating your confidence that its value is correct and fully accurate.
- If the value is obviously incorrect, then assign a score of 0 to this key.

Be strict when scoring:
- Do NOT give high scores for partially correct values, incomplete information, values which are missing but should not be, or content that should not be there.
- If you cannot verify the accuracy of a value, then your corresponding score should never exceed 50.
- Only assign high scores if this field in the proposed JSON Response is fully correct and complete.
- Any issue you identify or verification uncertainty should reduce your score.

## Output Format

Output a JSON object with the same top-level keys as the proposed JSON Response.
Each top-level key should map to an object with two fields:
- `explanation`: argue why the value in the proposed JSON Response might be incorrect
- `score`: confidence score between 0-100"""

    per_field_score_response_format: ClassVar[type[BaseModel]] = PerFieldCertaintyEvaluation

    @classmethod
    def create(cls, reasoning_effort: ReasoningEffort, **kwargs) -> ReflectionCompletionTemplate:
        if reasoning_effort == ReasoningEffort.NONE:
            raise ValueError("Per-field scoring only supports reasoning")

        return cls(
            prompt_template=cls._PROMPT,
            parse_patterns={},
            score_mapper=score_100_mapping,
            use_logprobs=False,
            per_field_score_key="score",
            **kwargs,
        )


SELF_REFLECTION_TEMPLATES_BY_WORKFLOW: dict[WorkflowType, list[type[ReflectionCompletionTemplate]]] = {
    WorkflowType.QA: [
        ReflectionCertaintyTemplate,
        ReflectionKnowledgeGapTemplate,
        ReflectionArgumentTemplate,
        ReflectionBinaryCorrectnessTemplate,
        ReflectionTrustworthinessTemplate,
        ReflectionCorrectnessTemplate,
    ],
    WorkflowType.CLASSIFICATION: [
        ReflectionClassificationCorrectnessTemplate,
        ReflectionClassificationScoringTemplate,
    ],
    WorkflowType.RAG: [
        ReflectionRAGCertaintyTemplate,
        ReflectionRAGArgumentTemplate,
        ReflectionRAGIssuesTemplate,
    ],
    WorkflowType.STRUCTURED_OUTPUT_SCORING: [
        ReflectionCertaintyTemplate,
        ReflectionKnowledgeGapTemplate,
        ReflectionArgumentTemplate,
        ReflectionSOPerScoreCorrectnessTemplate,
        ReflectionSOPerScoreCertaintyTemplate,
    ],
}
