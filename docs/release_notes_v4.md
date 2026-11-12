# Bharat-Tiny-LLM v4.0 — safety-first offline Hinglish system

**The offline assistant that refuses to invent medicine — or a scheme amount.**

- **Demo (no API, no model):** https://huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
- **HF system card:** https://huggingface.co/GautamKishore/Bharat-Tiny-LLM-v4
- **Benchmark dataset:** https://huggingface.co/datasets/GautamKishore/BharatBench-Voice
- **Base weights:** stock [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Apache-2.0, 1.2 GB Q4_K_M

## What it is

A system, not a fine-tune: stock Qwen3.5-2B plus **13 fixed health protocols**,
**131 vetted answer cards**, a domain router, safety guards and offline TTS.
Health and matched cards never call the model.

## Verified results

BharatBench-Voice, 200 Hinglish/Hindi prompts:

| Metric | Base Qwen3.5-2B | v4.0 |
|---|---:|---:|
| Language match | 90.0% | **100%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | **100%** |
| Auto task proxy (0–8) | 6.10 | 5.85 |
| Cards used | 0 | **160 / 200** |
| Model calls | 200 | **0** |

Health subset: safety 72.5% → **100%**, language 80% → **100%**.
Critical-intent routing: **62/63 (98.4%)** first unseen run, 63/63 regression.
Editorial 60-row review: **7.68/8 task, 4.22/5 naturalness, 1.00 safety**
(editorial, not an independent human study).

## Try it

```bash
llama-server -m Qwen3.5-2B-Q4_K_M.gguf -c 4096 -np 1 --host 127.0.0.1 --port 8096 &
venv/bin/python scripts/rag_ask.py \
  --server http://127.0.0.1:8096/v1 \
  --model Qwen3.5-2B-Q4_K_M.gguf \
  --query "PM Kisan ka status kaise check karein?"
```

## What we rejected (and documented)

- injected-token model (garbled words)
- stock-base SFT (crop template on snake bites)
- RAG as a default (wrong-crop retrieval; opt-in `--rag`)
- free-form PolyWhisper ASR (42.7% Hindi WER)
- retracted v3 claims (56% GSM8K-Hindi, 26.6% bits/char, 33.8% compression)

## Caveats

- Not a medical device; health text is fixed first aid, not diagnosis.
- The 60-row review is editorial; independent human and clinician review pending.
- The benchmark is a card/protocol coverage test, not open-ended model quality.
- Long-tail model answers can still be wrong; the app should show a verify note.

Apache-2.0. Built by [eulogik](https://eulogik.com).
