<div align="center">

# 🗣️ Bharat-Tiny-LLM v3

**A bilingual Hindi / Hinglish / English edge LLM that runs fully offline — 1.7B params, 1.0 GB GGUF, Apache-2.0.**

[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-eulogik%2FBharat--Tiny--LLM--v3-ff6f00)](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3)
[![HF Space](https://img.shields.io/badge/%F0%9F%9A%80%20Demo-In--Browser%20WebGPU-blue)](https://huggingface.co/spaces/eulogik/bharat-tiny-llm-v3-demo)
[![PyPI](https://img.shields.io/badge/PyPI-bharat--tiny--llm-3776ab)](https://pypi.org/project/bharat-tiny-llm/)
[![Params](https://img.shields.io/badge/Params-1.7B-orange)](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3)
[![GGUF](https://img.shields.io/badge/GGUF-Q4__K__M%20(1.0GB)-blue)](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/eulogik/Bharat-Tiny-LLM/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/eulogik/Bharat-Tiny-LLM?style=social)](https://github.com/eulogik/Bharat-Tiny-LLM)

</div>

> Built by [eulogik](https://eulogik.com) — an India-first AI lab shipping edge-deployable models.

> **🧪 v3.3 conversational** — 370 curated rows (evol-hindi + Hinglish + identity), r=8, 2 epochs, LR 5e-6. 46 steps, ~4 min on T4. Pushed as `brahmi-v3.3-f16.gguf` (3.4 GB) / `brahmi-v3.3-q4_k_m.gguf` (1.1 GB). Dataset: `eulogik/bharat-v3-chat-sft-v3`.

---

## ⚡ Try it (one command)

```bash
pip install "bharat-tiny-llm[mlx]" && bharat chat
```

Or open the **[in-browser WebGPU demo](https://huggingface.co/spaces/eulogik/bharat-tiny-llm-v3-demo)** — no install, fully private.

---

## 📌 What is Bharat-Tiny-LLM v3?

A **1.7-billion-parameter small language model (SLM)** for **Hindi, Hinglish, and English**
that runs on phones, laptops, and browsers — fully offline. Built by injecting 300
Devanagari subword tokens into **Qwen3-1.7B-Base**, warmup-training the new embeddings,
then capability **SFT** on Hindi math (GSM8K-Hindi) + Indic QA.

### Why it matters

| Capability | Base Qwen3-1.7B | Bharat-Tiny-LLM v3 |
|---|---:|---:|
| GSM8K-Hindi (grade-school math) | ~3% | **56%+** |
| Bits/char on held-out Hindi (↓) | 11.90 | **8.73** (−26.6%) |
| Devanagari token compression | — | **36.8%** |
| Edge-ready size (Q4_K_M) | — | **1.0 GB** |
| Runs offline | ❌ | ✅ |

---

## 🚀 Usage

### llama.cpp
```bash
llama-cli -m bharat-tiny-llm-v3-q4_k_m.gguf -p "नमस्ते, आप कैसे हैं?"
```

### Ollama
```bash
ollama create bharat-v3 -f Modelfile
ollama run bharat-v3
```

### Python
```python
from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="eulogik/Bharat-Tiny-LLM-v3",
    filename="bharat-tiny-llm-v3-q4_k_m.gguf",
)
print(llm("नमस्ते, आपका नाम क्या है?", max_tokens=128)["choices"][0]["text"])
```

---

## 📦 Model variants

| Repo | Format | Size | Use |
|---|---|---|---|
| [`Bharat-Tiny-LLM-v3`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3) | GGUF Q4_K_M | **1.0 GB** | **Edge / production (flagship)** |
| [`Bharat-Tiny-LLM-v3`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3) | GGUF F16 | 3.2 GB | Reference / fine-tuning |
| [`Bharat-Tiny-LLM-v2-MLX`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX) | Q4 MLX + LoRA | 880 MB | Legacy (Apple Silicon) |

---

## 🛠️ Training pipeline

```
corpus mining → inject 300 Devanagari tokens (mean-init)
  → embedding warmup (3000 steps, SGD+momentum, poison gate)
  → capability SFT (35K rows, LoRA r=32, DDP 2×T4)
  → merge + GGUF export
```

All training on consumer hardware (Kaggle 2×T4, ~6 GPU-hours for SFT). $0 cloud spend.

---

## 📊 Benchmarks

Full measured results: [docs/benchmarks_v2.md](docs/benchmarks_v2.md) (v3 section).

| Metric | v2 (Qwen2.5-1.5B) | v3 (Qwen3-1.7B) |
|---|---|---|
| GSM8K-Hindi | 0% | **56%+** |
| Bits/char (fair) | 1.49 (−10%) | **8.73 (−26.6%)** |
| Devanagari compression | 36.5% | 36.8% |
| Mixed corpus savings | 18.0% | 18.0% |

---

## 📜 License

Apache 2.0. Base model [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) (Apache 2.0).

## 📖 Citation

```bibtex
@misc{bharat-tiny-llm-v3,
  title={Bharat-Tiny-LLM v3: A Bilingual Edge LLM with Devanagari Token Injection and Hindi Reasoning},
  author={Eulogik},
  year={2026},
  url={https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3}
}
```

## 🔗 Links

- 🤗 Flagship model: https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3
- 🌐 Live demo: https://huggingface.co/spaces/eulogik/bharat-tiny-llm-v3-demo
- 📊 Benchmarks: [docs/benchmarks_v2.md](docs/benchmarks_v2.md)
- 💻 This repo: https://github.com/eulogik/Bharat-Tiny-LLM
- 🏢 Built by [eulogik](https://eulogik.com)
