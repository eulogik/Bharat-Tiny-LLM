# 🇮🇳 Bharat-Tiny-LLM v2 — Brahmi Token Injection

**300 injected Devanagari tokens → 36.5% Hindi token compression. An 880 MB offline Hindi/Hinglish LLM for Apple Silicon.**

<p align="center">
  <a href="https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX"><img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-eulogik%2FBharat--Tiny--LLM--v2--MLX-ff6f00"></a>
  <a href="https://pypi.org/project/bharat-tiny-llm/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-bharat--tiny--llm-3776ab"></a>
  <img src="https://img.shields.io/badge/License-Apache_2.0-green">
  <a href="https://eulogik.com"><img alt="Built by eulogik" src="https://img.shields.io/badge/Built%20by-eulogik-00b894"></a>
</p>

> Built by [eulogik](https://eulogik.com) — an India-first AI lab shipping edge-deployable models.

## ⚡ Try it (one command)

```bash
pip install "bharat-tiny-llm[mlx]" && bharat chat
```

The 880 MB model auto-downloads on first run, then works **fully offline**:

```
आप › Chai peete hain?
Chai peet hai. Ek line mein jawab deta hoon.
```

Commands: `bharat chat` (REPL), `bharat ask "सवाल"`, `bharat info`.

## ✨ What is Brahmi injection?

Qwen2.5's tokenizer spends ~1.04 tokens per Devanagari character. We mined the
436K-conversation corpus for high-frequency Devanagari subwords and **injected
the top 300** (+0.2% of the vocabulary) into the tokenizer, initializing each
new embedding as the mean of its constituent byte tokens. Two light training
phases recover quality: embedding-row warmup (3K steps) then LoRA (500 steps).

### Measured results ([full benchmarks](docs/benchmarks_v2.md))

| Metric | Base Qwen2.5-1.5B | Bharat-Tiny-LLM v2 |
|---|---|---|
| Tokens / 1000 Devanagari chars | 1041.9 | **661.6** (−36.5%) |
| Tokens / 1000 chars (mixed Hinglish+Hindi) | 593.3 | 486.6 (−18.0%) |
| Bits / char on held-out Hindi text ↓ | 1.6531 | **1.4878** (−10.0%) |
| Warmup val loss (injected rows) | 2.776 | **1.320** (−52.5%) |
| Model size (Q4 MLX) | — | 880 MB |

*We report both script-only and mixed-corpus compression: Hinglish/Latin spans
gain nothing from Devanagari tokens, so end-to-end savings depend on script mix.*

## 📦 Model variants

| Repo | Format | Size | Use |
|---|---|---|---|
| [`Bharat-Tiny-LLM-v2-MLX`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX) | Q4 MLX + LoRA | **880 MB** | Edge / Apple Silicon (**flagship**) |
| [`Bharat-Tiny-LLM-v2`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2) | PyTorch bf16 | 3.6 GB | Server / research |

## 🔧 Training pipeline

```
corpus mining → inject 300 tokens → mean-init embeddings
  → warmup (614K params, T4, 3K steps)
  → LoRA rank 8 (MLX M4, 500 steps ≈ 35 min)
  → 4-bit affine Q4 (group 64)
```

All training on consumer hardware: one Colab T4 + one Mac Mini M4. $0 cloud spend.

## 🚀 Python API

```python
from bharat_tiny_llm import chat

print(chat([{"role": "user", "content": "नमस्ते, आप कैसे हैं?"}],
           use_v2=True))
```

⚠️ **Known quirks:** don't pass a system prompt (v2 was trained without system
turns). Use the canonical sampling config (temp 0.3, top-p 0.85, rep-penalty
1.25) — the package and CLI do this by default.

## 📜 License

Apache 2.0. Base model [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (Apache 2.0).

## 📖 Citation

```bibtex
@techreport{kishore2026brahmilite,
    title={Brahmi-Lite: Minimal-Budget Devanagari Token Injection for Edge LLMs},
    author={Gautam Kishore},
    year={2026},
    institution={eulogik},
    url={https://github.com/eulogik/Bharat-Tiny-LLM}
}
```

## 🔗 Links

- 🤗 Flagship model: https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX
- 📊 Benchmarks: [docs/benchmarks_v2.md](docs/benchmarks_v2.md)
- 📄 Technical report: [docs/paper_arxiv.md](docs/paper_arxiv.md)
- 💻 This repo: https://github.com/eulogik/Bharat-Tiny-LLM
- 🏢 Built by [eulogik](https://eulogik.com)
