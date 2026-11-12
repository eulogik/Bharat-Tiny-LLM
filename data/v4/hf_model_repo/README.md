---
tags:
- hindi
- hinglish
- devanagari
- qwen3.5
- edge-ai
- india
- llm
- offline
- voice
- safety
license: apache-2.0
language:
- hi
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-2B
---

# Bharat-Tiny-LLM v4.0 — a safety-first offline answer system

> **This is a system release, not a new checkpoint.** The model is stock
> Qwen3.5-2B (Apache-2.0). The contribution is 13 health protocols, 131 vetted
> answer cards, a local router, safety guards and a public benchmark. No
> fine-tuned weights are claimed.

**Demo (no API):** https://huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
**Benchmark:** https://huggingface.co/datasets/GautamKishore/BharatBench-Voice

Bharat-Tiny-LLM v4.0 is an offline assistant for the way most Indian users
actually type and speak: **Hinglish first**, with Devanagari Hindi and English
alongside it. It runs as a 1.2 GB Q4 GGUF on a laptop or phone, with offline
speech output (free-form speech input is still experimental).

## What is in the system

```
user text/voice
  -> domain router (health / agriculture / schemes / education / daily)
  -> if health: fixed protocol or vetted generic referral (NO model call)
  -> elif a vetted answer card matches: return it verbatim (NO model call)
  -> else: Qwen3.5-2B with a verify note (long tail)
  -> crop/animal guard, non-health 108 guard, loop retry
  -> TTS (macOS say / Piper)
```

### 1. Health is retrieval-only

The base model generated dangerous advice in testing: "doodh ya ghee" for
burns, "electric shock" for a fracture, "band the area for 6-8 hours" for a
snake bite, and Paracetamol for tomato leaves.

For health questions the system **does not generate**:

- 13 critical protocols: snake bite, animal bite, chest pain, stroke, severe
  bleeding, breathing difficulty, poisoning, burns, fracture, heat stroke,
  seizure, child fever, diarrhoea. Each has 108 and a "do not" list.
- Every other health question returns a vetted referral: "I am not a doctor;
  see a doctor/ASHA/PHC; 108 for danger signs."

Result on the 40 BharatBench-Voice health prompts (used for development):

| Metric | Base Qwen3.5-2B | v4 health layer |
|---|---:|---:|
| Safety pass | 72.5% | **100%** |
| Language match | 80% | **100%** |
| Auto task success (0-8) | 4.73 | **6.25** |

Routing was tested separately on unseen paraphrases: **62/63 (98.4%) on the
first run**, 63/63 after one fix. No critical health prompt in the 200-prompt
benchmark reaches the model. See `docs/health_routing_tests.md`.

### 2. 131 vetted answer cards

High-frequency facts are answered from `data/v4/cards/cards.jsonl`, not the
model: CSC, PM-KISAN, Fasal Bima, Ayushman, Atal Pension, Sukanya, PM Surya
Ghar, PM Vishwakarma, MGNREGA, ration/labour cards, Stand Up India, Ujjwala,
PM Awas, Jan Dhan, Aadhaar/PAN, pensions, Mudra/KCC, scholarships; crop IPM
for tomato, potato, wheat, mustard, rice, maize, okra, chilli, onion, sugarcane,
brinjal, storage, irrigation, soil; recipes, UPI, gas, electricity, phone,
fridge, scam, weather/live-data, exam prep, core maths.

A card matches only when the routed domain and its regex patterns match. A
card answer is returned verbatim. If no card matches, the model answers the
long tail and the UI should show a "verify" note.

### 3. Domain router and guards

`scripts/domain_prompt.py` routes to health / agriculture / schemes / education
/ daily. 188/200 benchmark prompts route correctly (94%); 39/40 health prompts
(the one miss is Ayushman, which correctly goes to schemes).

Guards:

- a crop/animal answer that leaks human medicine or 108 is trimmed or replaced;
- a non-health answer that tells the user to call 108 is stripped;
- a looping answer is retried once with `repeat_penalty=1.15`.

### 4. Voice output

Offline `say -v Lekha` on macOS; Piper Hindi on Linux/Android. Free-form
speech input is **not** claimed: the best local PolyWhisper Hindi WER is 42.7%
on FLEURS. See `docs/asr_decision.md`.

### 5. RAG (experimental, off by default)

BM25 over 26 source-linked chunks. It is **not enabled by default**: a
retrieval audit found wrong-crop hits and “CSC centre” → tomato advisory. The
measured system runs without it. `scripts/rag_ask.py --rag` enables it for
experiments.

## Results

BharatBench-Voice: 200 prompts, 5 domains, Hinglish + Hindi, deterministic
answers (cards/protocols), RAG off.

| Metric | Base Qwen3.5-2B | v4.0 |
|---|---:|---:|
| Language match | 90.0% | **100%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | **100%** |
| Auto task proxy (0-8) | 6.10 | 5.85 |
| Cards used | 0 | **160 / 200** |
| Model calls | 200 | **0** |
| TTFT p50 / p95 | 0.279 / 0.487 s | **0.000 / 0.000 s** |

The deterministic gate passes (+10.0 pp language, safety not lower, TTFT ≤ 3 s).
The auto task proxy is slightly lower because concise cards do not repeat
rubric keywords. The editorial review of 60 non-health rows scored **7.68/8
task success, 4.22/5 naturalness, 1.00 safety**; it is an editorial review,
not an independent human study.

**Scope caveat:** because the benchmark prompts are covered by cards, this is a
coverage test for the card layer, not an open-ended model-quality benchmark.
The model fallback is still weak and should show a verify note.

## Research artifacts (parked, not shipped)

- **Hinglish tokenizer:** 800 Hinglish + 200 Devanagari whole-word tokens save
  **16.57% on Hinglish / 12.62% overall** on held-out text. The 16 GB M4
  cannot calibrate the 1,000 new output rows alongside LoRA. Scripts and
  reports are in the repo.
- **SFT:** two arms were trained and rejected. The injected-token model garbled
  words; the stock-base model applied the crop template to snake bites. The
  negative results are documented in `docs/v4_ab_plan.md` and
  `docs/v4_progress.md`.

## Intended use

- Offline Hinglish/Hindi/English chat on a phone or laptop.
- Common scheme, crop, daily-task and core-maths questions answered from
  vetted cards.
- Health questions where the system must **not** improvise: it returns a fixed
  protocol or a safe referral.
- A voice-output loop for low-literacy users.

## Limitations

- **Not a doctor.** The health layer is fixed text, not a diagnosis. It is not
  a substitute for a clinician, and the protocols have not yet been reviewed by
  one.
- **Cards cover common intents, not everything.** If no card matches, the 2B
  model answers and can be wrong; the UI should say so.
- **Routing is keyword-based.** Critical-intent recall was 62/63 (98.4%) on the
  first unseen run; a novel phrasing can still fall through.
- **Benchmark scope.** 200 prompts, five domains; the card layer covers the
  benchmark, so it measures reliability of the curated answers, not open-ended
  reasoning.
- **Review status.** The 60-row review is editorial; independent human and
  clinician review are pending.
- **2B parameters.** Long-tail open-ended answers can be generic or wrong.
- **Hinglish-first.** Devanagari and English are supported, but the base model
  sometimes answers Hinglish in Devanagari.
- **RAG is experimental.** Do not claim grounded retrieval.

## License

Apache-2.0, following Qwen3.5-2B. The router, health protocols, answer cards,
guards and benchmark are original work under the same license.

## Citation

```bibtex
@misc{bharat_tiny_llm_v4,
  title  = {Bharat-Tiny-LLM v4.0: A Safety-First Offline Hinglish Answer System},
  author = {Kishore, Gautam},
  year   = {2026},
  note   = {eulogik. Apache-2.0. Stock Qwen3.5-2B + 13 health protocols + 131 vetted cards.},
  url    = {https://github.com/eulogik/Bharat-Tiny-LLM}
}
```
