from typing import Dict, Set

# OpenAI Models
GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
GPT_4 = "gpt-4"
GPT_4_1 = "gpt-4.1"
GPT_4_1_MINI = "gpt-4.1-mini"
GPT_4_1_NANO = "gpt-4.1-nano"
GPT_4_5_PREVIEW = "gpt-4.5-preview"
GPT_4O = "gpt-4o"
GPT_4O_2024_11_20 = "gpt-4o-2024-11-20"
GPT_4O_MINI = "gpt-4o-mini"
GPT_5 = "gpt-5"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"
O1_PREVIEW = "o1-preview"
O1 = "o1"
O1_MINI = "o1-mini"
O3 = "o3"
O3_MINI = "o3-mini"
O4_MINI = "o4-mini"

# Bedrock Models
CLAUDE_3_HAIKU = "claude-3-haiku"
CLAUDE_3_5_HAIKU = "claude-3.5-haiku"
CLAUDE_3_SONNET = "claude-3-sonnet"
CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
CLAUDE_3_5_SONNET_V2 = "claude-3.5-sonnet-v2"
CLAUDE_3_7_SONNET = "claude-3.7-sonnet"
CLAUDE_OPUS_4 = "claude-opus-4-0"
CLAUDE_SONNET_4 = "claude-sonnet-4-0"
NOVA_MICRO = "nova-micro"
NOVA_LITE = "nova-lite"
NOVA_PRO = "nova-pro"

# Google Models
GEMINI_1_5_FLASH = "vertex_ai/gemini-1.5-flash"
GEMINI_1_5_PRO = "vertex_ai/gemini-1.5-pro"
GEMINI_2_0_FLASH_EXP = "vertex_ai/gemini-2.0-flash-exp"
GEMINI_FLASH_API = "gemini/gemini-flash-latest"
GEMINI_1_5_PRO_API = "gemini/gemini-1.5-pro-latest"
GEMINI_2_0_FLASH_API = "gemini/gemini-2.0-flash-exp"

# Azure Models
PHI_4 = "phi-4"

# DeepSeek Models
DEEPSEEK_CHAT = "deepseek/deepseek-chat"
DEEPSEEK_REASONER = "deepseek/deepseek-reasoner"

OPENAI_MODELS: Set[str] = {
    GPT_3_5_TURBO_16K,
    GPT_4,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    GPT_4_5_PREVIEW,
    GPT_4O,
    GPT_4O_MINI,
    GPT_5,
    GPT_5_MINI,
    GPT_5_NANO,
    O1_PREVIEW,
    O1,
    O1_MINI,
    O3,
    O3_MINI,
    O4_MINI,
}

BEDROCK_MODELS: Set[str] = {
    CLAUDE_3_HAIKU,
    CLAUDE_3_5_HAIKU,
    CLAUDE_3_SONNET,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_SONNET_V2,
    CLAUDE_3_7_SONNET,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    NOVA_MICRO,
    NOVA_LITE,
    NOVA_PRO,
}

BEDROCK_MODEL_TO_INFERENCE_PROFILE_ID: Dict[str, str] = {
    CLAUDE_3_HAIKU: "us.anthropic.claude-3-haiku-20240307-v1:0",
    CLAUDE_3_5_HAIKU: "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    CLAUDE_3_SONNET: "us.anthropic.claude-3-sonnet-20240229-v1:0",
    CLAUDE_3_5_SONNET: "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    CLAUDE_3_5_SONNET_V2: "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    CLAUDE_3_7_SONNET: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    CLAUDE_OPUS_4: "us.anthropic.claude-opus-4-20250514-v1:0",
    CLAUDE_SONNET_4: "global.anthropic.claude-sonnet-4-20250514-v1:0",
    NOVA_MICRO: "us.amazon.nova-micro-v1:0",
    NOVA_LITE: "us.amazon.nova-lite-v1:0",
    NOVA_PRO: "us.amazon.nova-pro-v1:0",
}

GOOGLE_MODELS: Set[str] = {
    # Vertex AI models (require ADC/service account)
    GEMINI_1_5_FLASH,
    GEMINI_1_5_PRO,
    GEMINI_2_0_FLASH_EXP,
    # Google AI Studio models (use with API keys)
    GEMINI_FLASH_API,
    GEMINI_1_5_PRO_API,
    GEMINI_2_0_FLASH_API,
}

AZURE_MODELS: Set[str] = {
    PHI_4,
}

DEEPSEEK_MODEL_PREFIX = "deepseek/"

# Requires API key from ai.google.dev
GEMINI_MODEL_PREFIX = "gemini/"
# Requires ADC/service account
VERTEX_AI_MODEL_PREFIX = "vertex_ai/"

REASONING_MODELS: Set[str] = {
    O1_PREVIEW,
    O1,
    O1_MINI,
    O3,
    O3_MINI,
    O4_MINI,
    GPT_5,
    GPT_5_MINI,
    GPT_5_NANO,
}


MODELS_WITH_LOGPROBS = {
    GPT_4,
    GPT_4O,
    GPT_4O_MINI,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    GPT_3_5_TURBO_16K,
}

DEFAULT_MODEL = GPT_4_1_MINI


# Encoding Models
CL100K_BASE = "cl100k_base"
O200K_BASE = "o200k_base"


ENCODING_MODELS: dict[str, str] = {
    DEFAULT_MODEL: O200K_BASE,
    GPT_3_5_TURBO_16K: CL100K_BASE,
    GPT_4: CL100K_BASE,
    GPT_4O: O200K_BASE,
    GPT_4O_MINI: O200K_BASE,
    O1_PREVIEW: O200K_BASE,
    O1: O200K_BASE,
    O1_MINI: O200K_BASE,
    O3_MINI: O200K_BASE,
    # NOTE: the claude-3/nova/default tokenizers are not correct, but currently it is only used to estimate token cost, so the approximate should be good enough.
    # If we start using logprobs for these models, this tokenizer has to be changed (there is no generally accepted claude tokenizer now)
    CLAUDE_3_HAIKU: CL100K_BASE,
    CLAUDE_3_SONNET: CL100K_BASE,
    CLAUDE_3_5_SONNET: CL100K_BASE,
    NOVA_MICRO: CL100K_BASE,
    NOVA_LITE: CL100K_BASE,
    NOVA_PRO: CL100K_BASE,
}
