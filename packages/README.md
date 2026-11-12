# 🇮🇳 bharat-tiny-llm — Offline Hindi/Hinglish AI for Apple Silicon

> **Version note (2026-11-11).** This package release (0.3.0) ships the **v2**
> MLX model (1.5B, 880 MB). The current model is **v3** (Qwen3-1.7B, 1.0 GB
> Q4_K_M GGUF). v3/v4 GGUF support lands in 0.4.0. See the
> [main README](https://github.com/eulogik/Bharat-Tiny-LLM) for current
> weights and corrected benchmarks.

**`bharat chat`** — an 880 MB Hindi/Hinglish assistant that runs fully offline. No API, no cloud, no internet.

[![PyPI version](https://img.shields.io/pypi/v/bharat-tiny-llm?color=3776ab)](https://pypi.org/project/bharat-tiny-llm/)
[![Python](https://img.shields.io/pypi/pyversions/bharat-tiny-llm)](https://pypi.org/project/bharat-tiny-llm/)
[![License](https://img.shields.io/pypi/l/bharat-tiny-llm?color=green)](https://github.com/eulogik/Bharat-Tiny-LLM)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-eulogik%2FBharat--Tiny--LLM--v2--MLX-ff6f00)](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX)

Built by [eulogik](https://eulogik.com).

## Install

```bash
pip install "bharat-tiny-llm[mlx]"   # Apple Silicon (recommended)
pip install "bharat-tiny-llm[torch]" # CPU/CUDA (loaders only)
```

## Use

```bash
bharat chat                  # interactive REPL with history
bharat ask "नमस्ते!"         # one-shot question
bharat info                  # model + package info
```

First run downloads the 880 MB model from Hugging Face; after that everything is offline.

```text
आप › Chai peete hain?
Chai peet hai, main ek assistant hoon.
```

### Python API

```python
from bharat_tiny_llm import chat

print(chat([{"role": "user", "content": "नमस्ते, आप कैसे हैं?"}], use_v2=True))
```

## What's inside

Qwen2.5-1.5B + **300 injected Devanagari tokens** ("Brahmi injection", −36.5%
Hindi tokenization cost) + LoRA, quantized to 4-bit MLX. Measured benchmarks:
[bharat-tiny-llm benchmarks](https://github.com/eulogik/Bharat-Tiny-LLM/blob/main/docs/benchmarks_v2.md).

## Notes & limitations

- Requires Apple Silicon for the CLI (`mlx`); Python `chat()` also works via the torch backend on other platforms.
- Don't pass system prompts — v2 was trained without system turns.
- Sampling defaults are canonical (temp 0.3, top-p 0.85, rep-penalty 1.25) — changing them can degrade output badly.
- Conversational quality is decent; factual/math reliability at 1.5B is limited.

## Links

- 🤗 Model: https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX
- 💻 Source: https://github.com/eulogik/Bharat-Tiny-LLM
- 📊 Benchmarks: https://github.com/eulogik/Bharat-Tiny-LLM/blob/main/docs/benchmarks_v2.md
- 🏢 Built by [eulogik](https://eulogik.com)

## License

Apache-2.0
