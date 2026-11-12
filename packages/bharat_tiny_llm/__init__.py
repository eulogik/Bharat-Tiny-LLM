"""Bharat-Tiny-LLM — India-first edge AI for Hinglish and Hindi.

The current PyPI release (0.3.0) packages the v2 MLX model. v3 is a 1.7B,
1.0 GB Q4_K_M GGUF on Hugging Face; v3/v4 GGUF support lands in 0.4.0.
Apache-2.0 licensed, open weights.

Built by `eulogik <https://eulogik.com>`_.
"""

from .constants import (
    MODEL_ID,
    MODEL_V2_ID,
    MODEL_V2_LORA,
    MODEL_V2_TORCH,
    FUSED_MODEL_ID,
    SPACE_ID,
    GITHUB_URL,
    BUILT_BY_URL,
    GENERATION_CONFIG,
)
from .loaders import load, chat

__version__ = "0.3.0"

__all__ = [
    "MODEL_ID",
    "MODEL_V2_ID",
    "MODEL_V2_LORA",
    "MODEL_V2_TORCH",
    "FUSED_MODEL_ID",
    "SPACE_ID",
    "GITHUB_URL",
    "BUILT_BY_URL",
    "GENERATION_CONFIG",
    "load",
    "chat",
]
