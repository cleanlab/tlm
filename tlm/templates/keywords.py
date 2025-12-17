from typing import Literal


PROMPT_PLACEHOLDER = "{prompt}"
QUESTION_PLACEHOLDER = "{question}"
ANSWER_PLACEHOLDER = "{answer}"
MAX_EXPLANATION_WORDS_PLACEHOLDER = "{max_explanation_words}"
CRITERIA_PLACEHOLDER = "{criteria}"
INPUT_1_PLACEHOLDER = "{input_1}"
INPUT_2_PLACEHOLDER = "{input_2}"

# Semantic evaluation placeholders
EVAL_CRITERIA_PLACEHOLDER = "{eval_criteria}"
QUERY_IDENTIFIER_PLACEHOLDER = "{query_identifier}"
CONTEXT_IDENTIFIER_PLACEHOLDER = "{context_identifier}"
RESPONSE_IDENTIFIER_PLACEHOLDER = "{response_identifier}"
QUERY_PLACEHOLDER = "{query}"
CONTEXT_PLACEHOLDER = "{context}"
REFERENCE_ANSWER_PLACEHOLDER = "{reference_answer}"


TemplateKeyword = Literal["prompt", "question", "answer", "max_explanation_words", "criteria", "input_1", "input_2"]
