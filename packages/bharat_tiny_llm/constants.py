"""Canonical identifiers and the recommended generation configuration."""

MODEL_ID = "eulogik/Bharat-Tiny-LLM"
FUSED_MODEL_ID = "eulogik/Bharat-Tiny-LLM-fused"
SPACE_ID = "eulogik/Bharat-Tiny-LLM"
GITHUB_URL = "https://github.com/eulogik/Bharat-Tiny-LLM"
HF_URL = "https://huggingface.co/eulogik/Bharat-Tiny-LLM"
SPACE_URL = "https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM"
BUILT_BY_URL = "https://eulogik.com"

# CRITICAL: the base Qwen2.5-1.5B emits garbled out-of-script tokens at high
# temperature. These parameters are required for clean, usable output.
GENERATION_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.85,
    "repetition_penalty": 1.25,
    "no_repeat_ngram_size": 3,
    "max_new_tokens": 256,
    "do_sample": True,
}
