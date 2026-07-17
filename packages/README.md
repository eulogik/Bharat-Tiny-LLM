# 🇮🇳 Bharat-Tiny-LLM

**India's first native edge AI for Hinglish & Hindi — running fully offline on ₹8,000 phones.**

`bharat-tiny-llm` is the official Python package for **Bharat-Tiny-LLM**: a 1.5B-parameter,
LoRA-fine-tuned language model that speaks fluent **Hinglish (Romanized Hindi)** and
**Devanagari Hindi**, and runs entirely on-device — no API, no cloud, no internet.

> Built by [eulogik](https://eulogik.com) · 🤗 Model: [eulogik/Bharat-Tiny-LLM](https://huggingface.co/eulogik/Bharat-Tiny-LLM) · 🚀 Live demo: [spaces/eulogik/Bharat-Tiny-LLM](https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM) · 💻 Code: [github.com/eulogik/Bharat-Tiny-LLM](https://github.com/eulogik/Bharat-Tiny-LLM)

---

## Why Bharat-Tiny-LLM?

- 🌐 **Truly bilingual** — Hinglish *and* Devanagari Hindi in one model (most Indic models do only one).
- 📱 **Edge-native** — 880 MB 4-bit build runs offline on ₹8,000 Android phones & Apple Silicon.
- 🆓 **Open & free** — Apache-2.0 weights, no vendor lock-in, self-hostable.
- 💸 **$0 training cost** — fine-tuned on a Mac Mini M4, zero cloud compute.
- 🪶 **Small** — 1.5B params, ~57 tok/s on a Mac Mini M4.

## Install

```bash
# Apple Silicon (recommended — MLX 4-bit, fastest)
pip install bharat-tiny-llm[mlx]

# Other platforms (CPU / CUDA, transformers backend)
pip install bharat-tiny-llm[torch]
```

## Quick start

```python
from bharat_tiny_llm import chat

reply = chat([
    {"role": "user", "content": "Chai peete hain?"},
])
print(reply)
```

The package ships with a **canonical generation config** (`temperature=0.3`,
`top_p=0.85`, `repetition_penalty=1.25`, `no_repeat_ngram_size=3`) so output is
clean out of the box — no garbled scripts, no degenerate loops. You normally
never have to tune these.

### Apple Silicon, low-level MLX

```python
from bharat_tiny_llm import load
from mlx_lm import generate

model, tokenizer = load()  # pulls eulogik/Bharat-Tiny-LLM (MLX 4-bit)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Biryani kaise banate hain?"}],
    tokenize=False, add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128))
```

## Model variants

| Repo | Format | Size | Use |
|------|--------|------|-----|
| [`eulogik/Bharat-Tiny-LLM`](https://huggingface.co/eulogik/Bharat-Tiny-LLM) | MLX 4-bit | ~880 MB | **Edge / Apple Silicon** (default) |
| [`eulogik/Bharat-Tiny-LLM-fused`](https://huggingface.co/eulogik/Bharat-Tiny-LLM-fused) | PyTorch fp16 | ~3.3 GB | Server / fine-tuning base |

## Links

- 🤗 Model card: https://huggingface.co/eulogik/Bharat-Tiny-LLM
- 🚀 Demo space: https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM
- 💻 Source: https://github.com/eulogik/Bharat-Tiny-LLM
- 🏢 Built by [eulogik](https://eulogik.com)

## License

Apache-2.0 (base Qwen2.5-1.5B weights Apache-2.0; LoRA adapter Apache-2.0).
