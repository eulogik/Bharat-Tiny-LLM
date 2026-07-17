"""Bharat-Tiny-LLM — India's first native edge AI for Hinglish and Hindi.

A 1.5B-parameter LoRA-fine-tuned language model that runs fully offline on
₹8,000 ($100) Android phones and Apple Silicon. Bilingual (Hinglish +
Devanagari Hindi), Apache-2.0 licensed, open weights.

Built by `eulogik <https://eulogik.com>`_.
"""

from .constants import (
    MODEL_ID,
    FUSED_MODEL_ID,
    SPACE_ID,
    GITHUB_URL,
    BUILT_BY_URL,
    GENERATION_CONFIG,
)
from .loaders import load, chat

__version__ = "0.1.0"

__all__ = [
    "MODEL_ID",
    "FUSED_MODEL_ID",
    "SPACE_ID",
    "GITHUB_URL",
    "BUILT_BY_URL",
    "GENERATION_CONFIG",
    "load",
    "chat",
]
