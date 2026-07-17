# 🇮🇳 Bharat-Tiny-LLM is live — India's first native edge AI for Hinglish & Hindi

Most "Indian language" models do **one** script. We built one that does **both** — and
runs fully offline on a ₹8,000 phone.

## What it is
Bharat-Tiny-LLM is a 1.5B-parameter, LoRA-fine-tuned model (base: Qwen2.5-1.5B) that
speaks fluent **Hinglish (Romanized Hindi)** *and* **Devanagari Hindi** from a single
model. No API. No cloud. No internet.

## Why it matters
- 🌐 **Truly bilingual** — Hinglish + Hindi, one 1.5B model
- 📱 **Edge-native** — 880 MB 4-bit build runs offline on ₹8,000 Android phones & Apple Silicon
- 💸 **$0 training cost** — fine-tuned on a Mac Mini M4, zero cloud compute
- 🆓 **Open weights** — Apache-2.0, self-hostable

## Try it
- 🤗 Model: https://huggingface.co/eulogik/Bharat-Tiny-LLM
- 🚀 Live demo: https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM
- 📦 `pip install bharat-tiny-llm`
- 💻 Code: https://github.com/eulogik/Bharat-Tiny-LLM

Built by @eulogik — India-first AI, shipped practical and edge-deployable. RT if you
want more open Indic models. 🧵👇

---

### Thread hooks (reply to above)
1/ We didn't train from scratch (4+ years). We warm-started Qwen2.5-1.5B and LoRA-tuned
on 436K cleaned Hinglish/Devanagari conversations. 110K iters on a Mac Mini M4.

2/ The hard part wasn't training — it was the "broken output" mystery. Turns out the base
model emits garbled Thai/CJK tokens at high temperature. Fix: temperature≈0.3 +
repetition_penalty 1.25 + n-gram blocking. Documented so you don't lose a week.

3/ On-device is the point. A 1.5B model at 4-bit fits in 880 MB and runs ~57 tok/s on a
Mac Mini M4. That's the footprint of a ₹8,000 phone, not a datacenter.

4/ Roadmap: more Indic languages (Tamil/Telugu/Bengali/Marathi), DPO preference tuning,
and BharatTiny-Bench — an eval suite for edge Indic models. Contributions welcome. 🚀
