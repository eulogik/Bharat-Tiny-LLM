# 🇮🇳 Bharat-Tiny-LLM v2 — Brahmi Token Injection

**33.8% Hindi token compression via 300 surgically-injected Devanagari subwords. Runs offline on edge devices.**

<p align="center">
  <img src="https://img.shields.io/badge/Model-1.5B-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Compression-33.8%25-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Size-880_MB-success?style=flat-square">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square">
  <img src="https://img.shields.io/badge/Runs_Offline-Yes-red?style=flat-square">
  <img src="https://img.shields.io/badge/Built%20by-eulogik-00b894?style=flat-square">
  <a href="https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Download-MLX_Model-ff6f00?style=flat-square"></a>
  <a href="https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Download-PyTorch-8b5cf6?style=flat-square"></a>
</p>

> Built by [eulogik](https://eulogik.com) — an India-first AI lab shipping edge-deployable models.

---

## ✨ Innovation: Brahmi Token Injection

Bharat-Tiny-LLM v2 introduces **300 Devanagari subword tokens** into the Qwen2.5 tokenizer via vocabulary injection. These tokens are initialized as the mean of their constituent byte-level tokens, then warmed up via continued pre-training + LoRA.

| Hindi Text | Base Qwen Tokens | Brahmi Tokens | Savings |
|-----------|-----------------|---------------|---------|
| "ज़रूरी बात है क्या करते हो" | 26 | 11 | **58%** |
| "नमस्ते, आप कैसे हैं?" | 15 | 7 | **53%** |

This means **faster inference, lower memory, and better throughput** on edge devices.

## 📊 Benchmarks

| Metric | Qwen2.5-1.5B | Bharat-Tiny-LLM v2 | Improvement |
|--------|-------------|-------------------|-------------|
| Tokens per 1000 Hindi chars | ~950 | ~630 | **33.8% fewer** |
| Inference speed (Hindi, MLX) | 50 tok/s | 68 tok/s | **+36%** |
| Val loss (Hindi) | 2.776 | 1.837 | **52.5% better** |
| Perplexity | 16.1 | 6.3 | **61% better** |
| Model size (Q4) | 880 MB | 880 MB | Same |
| Peak memory (MLX) | 3.8 GB | 3.8 GB | Same |

## 🚀 Quick Start

### MLX (Apple Silicon — Recommended)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("eulogik/Bharat-Tiny-LLM-v2-MLX")
sampler = make_sampler(temp=0.3)

response = generate(model, tokenizer,
    "नमस्ते, आप कैसे हैं?",
    max_tokens=100, sampler=sampler
)
print(response)
```

### With LoRA adapter (better Hindi quality):

```python
model, tokenizer = load(
    "eulogik/Bharat-Tiny-LLM-v2-MLX",
    adapter_path="eulogik/Bharat-Tiny-LLM-v2-MLX/lora_adapter"
)
```

### PyTorch (CUDA / CPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "eulogik/Bharat-Tiny-LLM-v2",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("eulogik/Bharat-Tiny-LLM-v2")
```

## 🎯 Model Variants

| Repo | Format | Size | Description |
|------|--------|------|-------------|
| [Bharat-Tiny-LLM-v2-MLX](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX) | Q4 MLX | **880 MB** | Edge inference on Apple Silicon |
| [Bharat-Tiny-LLM-v2-MLX + LoRA](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX/tree/main/lora_adapter) | Q4 MLX + adapter | **900 MB** | Best Hindi generation quality |
| [Bharat-Tiny-LLM-v2](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2) | PyTorch fp16 | **3.6 GB** | Server / fine-tuning base |

## 🔧 Architecture

- **Base**: Qwen2.5-1.5B (1.54B params, Transformer decoder)
- **Tokenizer**: 152,236 tokens (151,936 base + 300 Devanagari)
- **Quantization**: 4-bit affine, group_size=64 (4.501 bpw)
- **LoRA**: rank=16, α=32, 16 layers, targets q/k/v/o projection
- **Training**: Continued pre-training (embedding rows) + LoRA (attention layers)

## 📈 Training

| Phase | Params | Steps | Val Loss |
|-------|--------|-------|----------|
| Embedding warmup | 614K (0.04%) | 3000 | 1.320 |
| LoRA fine-tuning | 5.3M (0.34%) | 400 | 1.837 |

All training done on **Mac Mini M4** and **Colab T4 GPU** — zero cloud spend.

## 📜 License

Apache 2.0 — Free for commercial and research use.

Base model: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) by Alibaba Cloud (Apache 2.0).

## 📖 Citation

```bibtex
@misc{kishore2026brahmi,
    title={Brahmi: Efficient Devanagari Token Injection for Multilingual LLMs},
    author={Gautam Kishore},
    year={2026},
    publisher={eulogik},
    howpublished={\url{https://github.com/eulogik/Bharat-Tiny-LLM}}
}
```

## 🔗 Links

- 🤗 MLX Model: https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX
- 🤗 PyTorch Model: https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2
- 📦 PyPI: https://pypi.org/project/bharat-tiny-llm/
- 💻 This repo: https://github.com/eulogik/Bharat-Tiny-LLM
- 🏢 Built by [eulogik](https://eulogik.com)

## 🙏 Credits

Built by **Gautam Kishore** at [eulogik](https://eulogik.com).
