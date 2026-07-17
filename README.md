# 🇮🇳 Bharat-Tiny-LLM

**India's first native edge AI for Hinglish & Hindi — running fully offline on ₹8,000 phones.**

Bharat-Tiny-LLM is a 1.5B-parameter, LoRA-fine-tuned language model that speaks fluent
**Hinglish (Romanized Hindi)** and **Devanagari Hindi**, and runs entirely on-device.
No API. No cloud. No internet.

> Built by [eulogik](https://eulogik.com) — an India-first AI lab shipping practical, edge-deployable models.

<p align="center">
  <a href="https://huggingface.co/eulogik/Bharat-Tiny-LLM"><img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-eulogik%2FBharat--Tiny--LLM-ff6f00"></a>
  <a href="https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM"><img alt="Demo" src="https://img.shields.io/badge/%F0%9F%9A%80%20Demo-HF%20Space-8b5cf6"></a>
  <a href="https://pypi.org/project/bharat-tiny-llm/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-bharat--tiny--llm-3776ab"></a>
  <a href="https://eulogik.com"><img alt="Built by eulogik" src="https://img.shields.io/badge/Built%20by-eulogik-00b894"></a>
</p>

---

## Why

Most Indian-language models do **one** script. Bharat-Tiny-LLM does **both**, on hardware
that costs less than a pair of shoes:

- 🌐 **Truly bilingual** — Hinglish *and* Devanagari Hindi in one 1.5B model.
- 📱 **Edge-native** — 880 MB 4-bit build runs offline on ₹8,000 Android phones & Apple Silicon.
- 💸 **$0 training cost** — fine-tuned on a Mac Mini M4, zero cloud compute.
- 🆓 **Open weights** — Apache-2.0, self-hostable, no lock-in.

## Results

| Metric | Value |
|--------|-------|
| Base model | [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (multilingual, 29 languages) |
| Training data | **436K** cleaned Hinglish/Devanagari conversations |
| Method | LoRA (16 layers, rank 16, alpha 32, scale 2.0) |
| Training iters | 110,000 |
| Best val loss | **0.937** |
| Q4 model size | **880 MB** (MLX 4-bit) |
| Inference speed | ~57 tok/s on Mac Mini M4 |
| Training cost | $0 cloud compute |
| License | Apache-2.0 |

## Quick start

```bash
# Apple Silicon (recommended — MLX 4-bit)
pip install bharat-tiny-llm[mlx]

# CPU / CUDA
pip install bharat-tiny-llm[torch]
```

```python
from bharat_tiny_llm import chat

reply = chat([{"role": "user", "content": "Chai peete hain?"}])
print(reply)
```

The package ships with a **canonical generation config** (`temperature=0.3`,
`top_p=0.85`, `repetition_penalty=1.25`, `no_repeat_ngram_size=3`) so output is
clean out of the box — no garbled scripts, no degenerate loops.

### MLX CLI

```bash
pip install mlx-lm
mlx_lm chat --model eulogik/Bharat-Tiny-LLM
```

## Model variants

| Repo | Format | Size | Use |
|------|--------|------|-----|
| [`eulogik/Bharat-Tiny-LLM`](https://huggingface.co/eulogik/Bharat-Tiny-LLM) | MLX 4-bit | ~880 MB | **Edge / Apple Silicon** |
| [`eulogik/Bharat-Tiny-LLM-fused`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-fused) | PyTorch fp16 | ~3.3 GB | Server / fine-tuning base |

## Links

- 🤗 Model: https://huggingface.co/eulogik/Bharat-Tiny-LLM
- 🚀 Demo: https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM
- 📦 PyPI: https://pypi.org/project/bharat-tiny-llm/
- 💻 This repo: https://github.com/eulogik/Bharat-Tiny-LLM
- 🏢 Built by [eulogik](https://eulogik.com)

## License

Apache-2.0 (base Qwen2.5-1.5B weights Apache-2.0; LoRA adapter Apache-2.0).

---

### Repo layout

```
Bharmi/                      # training & experiments
├── data/processed/          # cleaned training conversations
├── scripts/                 # training, fusion, data-cleaning scripts
├── configs/                 # generation & training configs
├── packages/                # the bharat-tiny-llm PyPI package source
├── living.md                # full project walkthrough
└── README.md                # this file
```
