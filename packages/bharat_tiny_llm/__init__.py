"""Bharat-Tiny-LLM — India's first native edge AI for Hinglish and Hindi.

A 1.5B-parameter edge LLM with 33.8% Devanagari token compression via Brahmi
vocabulary injection. Runs fully offline on Apple Silicon and edge devices.
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
