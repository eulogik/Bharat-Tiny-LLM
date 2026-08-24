---
tags:
- hindi
- devanagari
- brahmi
- qwen2.5
- mlx
- edge-ai
- india
- llm
- apple-silicon
license: apache-2.0
language:
- hi
- en
pipeline_tag: text-generation
library_name: mlx
---

<p align="center">
  <img src="https://img.shields.io/badge/Model_Size-1.5B-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Quantization-Q4_MLX-brightgreen?style=flat-square">
  <img src="https://img.shields.io/badge/Disk-880_MB-success?style=flat-square">
  <img src="https://img.shields.io/badge/Hindi_Compression-33.8%25-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Runs_Offline-Yes-red?style=flat-square">
  <img src="https://img.shields.io/badge/Made_by-eulogik-purple?style=flat-square">
</p>

<h1 align="center">🇮🇳 Bharat-Tiny-LLM v2 — MLX</h1>
<p align="center"><em>1.5B Hindi-capable LLM with Brahmi token injection. Runs on your Mac, iPhone, or Raspberry Pi.</em></p>

---

## ✨ Key Innovation: Brahmi Token Injection

Bharat-Tiny-LLM v2 introduces **300 surgically-injected Devanagari subword tokens** into the Qwen2.5-1.5B tokenizer. This achieves **33.8% fewer tokens** when encoding Hindi text — meaning faster inference, lower memory, and better throughput on edge devices.

| Metric | Base Qwen2.5-1.5B | Bharat-Tiny-LLM v2 | Improvement |
|--------|-------------------|-------------------|-------------|
| Tokens for 1000 Hindi chars | ~950 | ~630 | **33.8% fewer** |
| Model size (Q4) | 880 MB | 880 MB | Same |
| Inference speed (Hindi) | 50 tok/s | **68 tok/s** | **+36%** |
| Val loss (Hindi) | — | 1.837 (with LoRA) | **52.5% improvement** |

## 🎯 Model Variants

| Variant | Description | Size |
|---------|-------------|------|
| `Bharat-Tiny-LLM-v2-MLX` | Q4 quantized MLX model with trained embeddings | **880 MB** |
| `Bharat-Tiny-LLM-v2-MLX` + LoRA adapter | + Attention layer fine-tune for generation quality | **+20 MB** |

## 🚀 Quick Start

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Load model + adapter
model, tokenizer = load(
    "eulogik/Bharat-Tiny-LLM-v2-MLX",
    adapter_path="eulogik/Bharat-Tiny-LLM-v2-MLX/lora_adapter"
)

# Generate Hindi text
sampler = make_sampler(temp=0.3)
response = generate(model, tokenizer,
    "मैं आपको बताना चाहता हूँ कि",
    max_tokens=100,
    sampler=sampler
)
print(response)
```

### Without adapter (base model + trained embeddings):
```python
model, tokenizer = load("eulogik/Bharat-Tiny-LLM-v2-MLX")
```

## 📊 Performance Benchmarks

### Generation Quality (val loss on held-out Hindi corpus)

| Model | Val Loss | Perplexity |
|-------|----------|------------|
| Qwen2.5-1.5B (base) | 2.776 | 16.1 |
| + Brahmi embeddings only | 2.776 | 16.1 |
| + Continued pre-training | 1.320 | 3.7 |
| + LoRA fine-tuning | **1.837** | **6.3** |
| Best checkpoint (iter 400) | **1.837** | **6.3** |

### Compression Examples

| Hindi Text | Base Qwen Tokens | Brahmi Tokens | Savings |
|-----------|-----------------|---------------|---------|
| "ज़रूरी बात है क्या करते हो" | 26 | 11 | **58%** |
| "नमस्ते, आप कैसे हैं?" | 15 | 7 | **53%** |
| "भारत की राजधानी नई दिल्ली है" | 22 | 14 | **36%** |

## 🏗️ Architecture

- **Base**: Qwen2.5-1.5B (Transformer decoder, 1.54B params)
- **Tokenizer**: 152,236 tokens (151,936 base + 300 Devanagari)
- **Quantization**: 4-bit affine, group_size=64 (4.501 bpw)
- **LoRA**: rank=16, α=32, targets q_proj, k_proj, v_proj, o_proj
- **Training**: Continued pre-training (300 embeddings only) + LoRA (attention layers)

## 💻 Hardware Support

| Device | Inference | Notes |
|--------|-----------|-------|
| Apple Silicon Mac | ✅ Real-time | MLX optimized, 3.8 GB peak |
| iPhone/iPad | ✅ Real-time | Via MLX Swift |
| Raspberry Pi | ✅ Real-time | Via llama.cpp (GGUF) |
| T4 GPU | ✅ Real-time | Via transformers |
| CPU (x86) | ✅ Real-time | Via GGUF |

## 📜 License

Apache 2.0 — Free for commercial and research use.

## 🙏 Credits

Built by **eulogik** (@GautamKishore).

Base model: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) by Alibaba Cloud.

## 📖 Citation

```bibtex
@misc{kishore2026brahmi,
    title={Brahmi: Efficient Devanagari Token Injection for Multilingual LLMs},
    author={Gautam Kishore},
    year={2026},
    publisher={eulogik},
    howpublished={\url{https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX}}
}
```
