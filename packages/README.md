# 🇮🇳 Bharat-Tiny-LLM — Edge AI for Hindi & Hinglish

**33.8% Hindi token compression via Brahmi vocabulary injection. Runs offline on Apple Silicon.**

[![PyPI version](https://img.shields.io/pypi/v/bharat-tiny-llm?color=3776ab)](https://pypi.org/project/bharat-tiny-llm/)
[![Python](https://img.shields.io/pypi/pyversions/bharat-tiny-llm)](https://pypi.org/project/bharat-tiny-llm/)
[![License](https://img.shields.io/pypi/l/bharat-tiny-llm?color=green)](https://github.com/eulogik/Bharat-Tiny-LLM)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-eulogik%2FBharat--Tiny--LLM--v2--MLX-ff6f00)](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX)

`bharat-tiny-llm` is the official Python package for **Bharat-Tiny-LLM v2**: a 1.5B-parameter LLM fine-tuned for **Hinglish** and **Devanagari Hindi**, with **33.8% token compression** achieved via Brahmi script injection — 300 Devanagari subwords injected into Qwen2.5's tokenizer.

> Built by [eulogik](https://eulogik.com) — an India-first AI lab shipping edge AI.

---

## ✨ v0.2.0 — Brahmi Token Injection

| Feature | Description |
|---------|-------------|
| **Brahmi injection** | 300 Devanagari subwords → 33.8% fewer tokens for Hindi |
| **v2 model** | `eulogik/Bharat-Tiny-LLM-v2-MLX` with LoRA adapter |
| **52.5% loss improvement** | Vs base Qwen2.5-1.5B on Hindi text |
| **880 MB** | Q4 MLX, runs offline on Mac/iPhone/iPad |

## Install

```bash
# Apple Silicon (recommended — MLX, fastest)
pip install bharat-tiny-llm[mlx]

# Other platforms (CPU / CUDA, transformers)
pip install bharat-tiny-llm[torch]
```

## Quick start

### v2 model (recommended)

```python
from bharat_tiny_llm import chat

# Uses the v2 MLX model with LoRA adapter
reply = chat(
    [{"role": "user", "content": "कितने बजे मिलना है?"}],
    use_v2=True,
    adapter_path="eulogik/Bharat-Tiny-LLM-v2-MLX/lora_adapter",
)
print(reply)
```

### v1 model (legacy)

```python
from bharat_tiny_llm import chat

reply = chat([{"role": "user", "content": "Chai peete hain?"}])
print(reply)
```

### Low-level MLX

```python
from bharat_tiny_llm import load
from mlx_lm import generate

model, tokenizer = load(use_v2=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "नमस्ते, आप कैसे हैं?"}],
    tokenize=False, add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128))
```

## Model variants

| Repo | Format | Size | Description |
|------|--------|------|-------------|
| `eulogik/Bharat-Tiny-LLM-v2-MLX` | Q4 MLX | 880 MB | **v2 edge model (recommended)** |
| `eulogik/Bharat-Tiny-LLM-v2` | PyTorch fp16 | 3.6 GB | v2 for server/fine-tuning |
| `eulogik/Bharat-Tiny-LLM` | Q4 MLX | 880 MB | v1 edge model (legacy) |
| `eulogik/Bharat-Tiny-LLM-fused` | PyTorch fp16 | 3.3 GB | v1 for server/fine-tuning |

## License

Apache-2.0

## Links

- 🤗 Model: https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX
- 💻 Source: https://github.com/eulogik/Bharat-Tiny-LLM
- 🏢 Built by [eulogik](https://eulogik.com)
