import re

from tlm.types import ExtractedResponseField, RegexPattern

RESPONSE_PARSER = {
    ExtractedResponseField.ANSWER: [
        RegexPattern(
            regex=[
                r"Response:\s*\[([^\]]+)\]",
                r"Response:\s*([^\]]+)",
                r"\s*\[([^\]]+)\]",
                r"^(.+)$",  # Fallback if no format matches
            ]
        ),
    ],
}

REASONING_RESPONSE_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r"Reasoning:\s*\[([^\]]+)\](?:,\s*Response:\s*\[[^\]]+\])",
                r"Reasoning:\s*(.*?)(?:\s*Response:\s*.+)",
            ]
        ),
    ],
    **RESPONSE_PARSER,
}

SCORE_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r"<score>\s*(.*?)\s*</score>",
                r".*<score>\W*(?P<answer>\d+(?:\.\d+)?)\W*</score>",
                r".*\(?(?P<answer>\d+(?:\.\d+)?)\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}

THINK_SCORE_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r"<think>\s*(.*?)\s*</think>\s*<score>\s*(?:.*?)\s*</score>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
    **SCORE_XML_PARSER,
}

RATING_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*<rating>\W*(?P<answer>one|two|three|four|five|[12345])\W*</rating>",
                r".*<rating>\W*(?P<answer>[12345])\W*</rating>",
                r".*\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r"<rating>\s*(.*?)\s*</rating>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
        RegexPattern(
            regex=[
                r"\s*(.*?)\s*<rating>\s*(.*?)\s*</rating>",
            ]
        ),
    ],
}

THINK_RATING_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r"<think>\s*(.*?)\s*</think>\s*<rating>\s*(?:.*)\s*</rating>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
    **RATING_XML_PARSER,
}

CHOICE_TRUE_FALSE_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r"<choice>\s*(.*?)\s*</choice>",
                r".*<choice>\s*((?P<answer>True|False))\s*</choice>",
                r".*<choice>\W*((?P<answer>True|False))\W*</choice>",
                r".*\(((?P<answer>True|False))\)",
                r".*\(?((?P<answer>True|False))\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}

THINK_CHOICE_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r"<think>\s*(.*?)\s*</think>\s*<choice>\s*(?:.*?)\s*</choice>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
    **CHOICE_TRUE_FALSE_XML_PARSER,
}

RATING_10_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*<rating>\W*(?P<answer>10|[0-9]|zero|one|two|three|four|five|six|seven|eight|nine|ten)\W*</rating>",
                r".*<rating>\W*(?P<answer>10|[0-9])\W*</rating>",
                r".*\(?(?P<answer>10|[0-9]|zero|one|two|three|four|five|six|seven|eight|nine|ten)\)?",
                r"<rating>\s*(.*?)\s*</rating>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}

THINK_RATING_10_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r"<think>\s*(.*?)\s*</think>\s*<rating>\s*(?:.*)\s*</rating>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
    **RATING_10_XML_PARSER,
}

DECISION_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*<decision>\W*(?P<answer>one|two|three|four|five|[12345])\W*</decision>",
                r".*<decision>\W*(?P<answer>[12345])\W*</decision>",
                r".*\(?(?P<answer>one|two|three|four|five|[12345])\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}

THINKING_DECISION_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r"<thinking>\s*(.*?)\s*</thinking>\s*<decision>\s*(?:.*?)\s*</decision>",
            ]
        )
    ],
    **DECISION_XML_PARSER,
}

CHOICE_A_B_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r"Choice: \[(?P<choice>[ABab])\]",
                r"Choice: (?P<choice>[ABab])",
            ],
        ),
    ],
}

ANSWER_YES_NO_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*<answer>\s*(Yes|No)\s*</answer>",
                r".*<answer>\W*(Yes|No)\W*</answer>",
                r".*\((Yes|No)\)",
                r".*\(?(Yes|No)\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        )
    ],
}

ANSWER_XML_PARSER = {
    ExtractedResponseField.ANSWER: [
        RegexPattern(
            regex=[
                r".*<answer>\W*(?P<answer>.*)\W*</answer>",
                r"(?P<answer>.*)",
            ],
        ),
    ],
}

THINK_ANSWER_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=r"<think>\s*(.*?)\s*</think>\s*<answer>\s*(?:.*?)\s*</answer>",
        ),
    ],
    **ANSWER_XML_PARSER,
}

CHOICE_YES_NO_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*<choice>\s*(?P<answer>Yes|No)\s*</choice>",
                r".*<choice>\W*(?P<answer>Yes|No)\W*</choice>",
                r".*\((?P<answer>Yes|No)\)",
                r".*\(?(?P<answer>Yes|No)\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        )
    ],
}

SCORE_10_XML_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*<score>\W*(?P<answer>10|[0-9]|zero|one|two|three|four|five|six|seven|eight|nine|ten)\W*</score>",
                r".*<score>\W*(?P<answer>10|[0-9])\W*</score>",
                r".*\(?(?P<answer>10|[0-9]|zero|one|two|three|four|five|six|seven|eight|nine|ten)\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}

ISSUES_SCORE_XML_PARSER = {
    ExtractedResponseField.EXPLANATION: [
        RegexPattern(
            regex=[
                r".*<issues>\W*(?P<explanation>.*)\W*</issues>.*?<score>\s*(?:.*?)\s*</score>",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
    **SCORE_XML_PARSER,
}

CHOICE_AB_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*\n Choice:\W\(?(?P<answer>[AB])\)?",
                r".*[\n]? Choice:\W\(?(?P<answer>[AB])\)?",
                r".*\n[ ]?Choice:\W\(?(?P<answer>[AB])\)?",
                r".*[\n]+[ ]*Choice[:]?\W\(?(?P<answer>[AB])\)?",
                r".*[\n]*[ ]+Choice\W?\(?(?P<answer>[AB])\)?",
            ],
            flags=re.DOTALL,
        ),
        RegexPattern(
            regex=[
                r".*\n choice:\W\(?(?P<answer>[ab])\)?",
                r".*[\n]? choice:\W\(?(?P<answer>[ab])\)?",
                r".*\n[ ]?choice:\W\(?(?P<answer>[ab])\)?",
                r".*[\n]?[ ]?choice[:]?\W\(?(?P<answer>[ab])\)?",
                r".*choice:\W\(?(?P<answer>[ab])\)?",
                r".*choice\W\(?(?P<answer>[ab])\)?",
                r".*\(?(?P<answer>[A])\)?[ ]?Correct",
                r".*\(?(?P<answer>[B])\)?[ ]?Incorrect",
                r".*\((?P<answer>[ab])\)",
                r".*\(?(?P<answer>[ab])\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}

RATING_1_5_PARSER = {
    ExtractedResponseField.SCORE: [
        RegexPattern(
            regex=[
                r".*\n Rating:\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*[\n]? Rating:\W\(?(?P<answer>[12345])\)?",
                r".*\n[ ]?Rating:\W\(?(?P<answer>[12345])\)?",
                r".*[\n]+[ ]*Rating[:]?\W\(?(?P<answer>[12345])\)?",
                r".*[\n]*[ ]+Rating\W?\(?(?P<answer>one|two|three|four|five|[12345])\)?",
            ],
            flags=re.DOTALL,
        ),
        RegexPattern(
            regex=[
                r".*\n rating:\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*[\n]? rating:\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*\n[ ]?rating:\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*[\n]?[ ]?rating[:]?\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*rating:\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*rating\W\(?(?P<answer>one|two|three|four|five|[12345])\)?",
                r".*\(?(?P<answer>one|two|three|four|five|[12345])\)?",
            ],
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ],
}
