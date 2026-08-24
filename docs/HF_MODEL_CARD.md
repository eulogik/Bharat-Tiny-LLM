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
  <img src="https://img.shields.io/badge/Params-1.5B-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Quantization-Q4_MLX-brightgreen?style=flat-square">
  <img src="https://img.shields.io/badge/Disk-880_MB-success?style=flat-square">
  <img src="https://img.shields.io/badge/Devanagari_Compression-36.5%25-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Runs_Offline-Yes-red?style=flat-square">
  <img src="https://img.shields.io/badge/Made_by-eulogik-purple?style=flat-square">
</p>

<h1 align="center">🇮🇳 Bharat-Tiny-LLM v2 — MLX</h1>
<p align="center"><em>1.5B Hindi/Hinglish LLM with Brahmi token injection. 880 MB, fully offline on Apple Silicon.</em></p>

---

## ✨ What is Brahmi injection?

300 high-frequency Devanagari subword tokens injected into the Qwen2.5 tokenizer
(+0.2% vocab), embeddings initialized as the mean of constituent byte tokens,
recovered with embedding warmup + LoRA.

### Measured benchmarks (held-out data; [methodology](https://github.com/eulogik/Bharat-Tiny-LLM/blob/main/docs/benchmarks_v2.md))

| Metric | Base Qwen2.5-1.5B | v2 | Δ |
|---|---|---|---|
| Tokens / 1000 Devanagari chars | 1041.9 | 661.6 | **−36.5%** |
| Tokens / 1000 chars (mixed Hinglish+Hindi) | 593.3 | 486.6 | −18.0% |
| Bits / char (held-out Hindi, fair protocol) | 1.6531 | 1.4878 | **−10.0%** |
| Warmup val loss (injected rows) | 2.776 | 1.320 | −52.5% |

*End-to-end savings are lower than script-only because Latin/Hinglish spans gain
nothing from Devanagari tokens.*

## 🚀 Quick Start

### The easy way — `bharat` CLI

```bash
pip install "bharat-tiny-llm[mlx]"
bharat chat        # interactive Hindi/Hinglish REPL, works offline
bharat ask "नमस्ते!"
```

### Python

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load(
    "eulogik/Bharat-Tiny-LLM-v2-MLX",
    adapter_path="eulogik/Bharat-Tiny-LLM-v2-MLX",  # see note below
)
```

The LoRA adapter ships in the `lora_adapter/` folder of this repo. `mlx_lm.load`
needs a local directory for adapters — easiest is:

```python
from huggingface_hub import snapshot_download
adir = snapshot_download("eulogik/Bharat-Tiny-LLM-v2-MLX",
                         allow_patterns=["lora_adapter/*"]) + "/lora_adapter"
model, tokenizer = load("eulogik/Bharat-Tiny-LLM-v2-MLX", adapter_path=adir)

sampler = make_sampler(temp=0.3)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Chai peete hain?"}],
    tokenize=False, add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128, sampler=sampler))
```

## ⚠️ Important usage notes

1. **No system prompts** — v2 was trained without system turns; adding one degrades output.
2. **Use temp ≤ 0.3** with top-p 0.85 and repetition penalty ≥ 1.2. Higher temperatures produce garbled out-of-script tokens.
3. **Adapter required** — the base expanded model without the LoRA adapter generates poorly.
4. Adapter fixed Aug 24, 2026: if you downloaded earlier, re-pull `lora_adapter/`.

## 🎯 Training

| Phase | Params | Hardware | Result |
|---|---|---|---|
| Embedding warmup (300 rows) | 614K (0.04%) | Colab T4, 3K steps | val 2.776 → 1.320 |
| LoRA (rank 8, scale 20, 16 layers) | ~5M | Mac Mini M4, 500 steps ≈ 35 min | val 1.837 @ step 400 |

## 📜 License & attribution

Apache 2.0. Base: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) © Alibaba Cloud, Apache 2.0.
Built by [eulogik](https://eulogik.com) · [GitHub](https://github.com/eulogik/Bharat-Tiny-LLM) · [PyPI](https://pypi.org/project/bharat-tiny-llm/)

## Citation

```bibtex
@techreport{kishore2026brahmilite,
    title={Brahmi-Lite: Minimal-Budget Devanagari Token Injection for Edge LLMs},
    author={Gautam Kishore},
    year={2026},
    institution={eulogik},
    url={https://github.com/eulogik/Bharat-Tiny-LLM}
}
```
