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

# Bharat-Tiny-LLM v4 — a safety-first offline Hinglish system

> **This is a system release, not a new checkpoint.** The model is stock
> Qwen3.5-2B (Apache-2.0). The contribution is the local safety/router/RAG
> layer and the public benchmark. No fine-tuned weights are claimed.

Bharat-Tiny-LLM v4 is an offline assistant for the way most Indian users
actually type and speak: **Hinglish first**, with Devanagari Hindi and English
alongside it. It runs as a 1.2 GB Q4 GGUF on a laptop or a phone, with a
planned on-device voice loop (speech output now; free-form speech input is
still experimental).

## What is in the system

```
user text/voice
  -> domain router (health / agriculture / schemes / education / daily)
  -> if health: fixed protocol or vetted generic response (NO model call)
  -> else: BM25 RAG over 26 source-linked chunks + Qwen3.5-2B
  -> TTS (macOS say / Piper)
```

### 1. Health is retrieval-only

The base model generated dangerous advice in testing: "doodh ya ghee" for
burns, "electric shock" for a fracture, "band the area for 6-8 hours" for a
snake bite, and Paracetamol for tomato leaves. A 2B model cannot be trusted to
improvise first aid.

For health questions the system **does not generate**:

- Critical intents (snake bite, animal bite, chest pain, stroke, severe
  bleeding, breathing difficulty, poisoning, burns, fracture, heat stroke,
  seizure, child fever, diarrhoea) return a fixed, vetted protocol with 108 and
  the "do not" list.
- Every other health question returns a vetted generic response: "I am not a
  doctor; see a doctor/ASHA/PHC; 108 for danger signs."

Result on the 40 BharatBench-Voice health prompts (used for development):

| Metric | Base Qwen3.5-2B | v4 health layer |
|---|---:|---:|
| Safety pass | 72.5% | **100%** |
| Language match | 80% | **100%** |
| Junk-free | 100% | 100% |
| Auto task success (0-8) | 4.73 | **6.15** |

Because those 40 prompts were used to tune the matcher, routing was tested
separately on unseen paraphrases, misspellings and Devanagari. Critical-intent
routing scored **62/63 (98.4%) on the first run of the final unseen set**; after
one fix the regression suite is 63/63. No critical health prompt in the
200-prompt benchmark is routed to the model. The honest public claim is 98.4%,
not 100%, on unseen phrasing. See `docs/health_routing_tests.md`.

### 2. Domain router

A keyword router (`scripts/domain_prompt.py`) selects the per-domain system
prompt and keeps human health, animal health, crops and schemes apart.
188/200 BharatBench-Voice prompts route to the expected domain (94%),
including 39/40 health prompts; the one miss is Ayushman, which correctly goes
to schemes.

### 3. RAG

BM25 over 26 curated, source-linked chunks: PM Kisan, Ayushman, Fasal Bima,
Mudra, PM Awas, Ujjwala, Jan Dhan, ration, MGNREGA, KCC, Soil Health Card,
Sukanya, Atal Pension, e-Shram, Aadhaar, crop advisories, health safety, and a
live-data policy. Hinglish/Devanagari query expansion. No embeddings, no
network, phone-class CPU.

### 4. Voice output

Offline `say -v Lekha` on macOS; Piper Hindi on Linux/Android. Free-form
speech input is **not** claimed: the best local PolyWhisper Hindi WER is 42.7%
on FLEURS, which is too high for unsupervised use. See `docs/asr_decision.md`.

## Results

BharatBench-Voice: 200 prompts, 5 domains, Hinglish + Hindi, temperature 0,
`enable_thinking=false`, non-streaming, no draft model.

| Metric | Base Qwen3.5-2B | v4 system |
|---|---:|---:|
| Language match | 89.5% | **95.0%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | 100% |
| Loop-free (5-gram < 0.50) | 99.5% | **100%** |
| Auto task success (0-8) | 6.08 | **6.59** |
| TTFT p50 / p95 | 0.279 / 0.487 s | 0.229 / 0.339 s |
| Total p50 / p95 | 9.10 / 17.47 s | 6.43 / 6.88 s |

The full-system (router + health layer + RAG) run is in
`eval/bharatbench_voice/runs/system_full.jsonl`. The health subset is 100%
safety; the non-health subset uses the base model and inherits its strengths
and weaknesses.

**Near-duplicate caveat:** 36/200 benchmark prompts share ≥0.60 Jaccard with an
SFT template prompt (16/40 agriculture). The full-set score measures safety
template learning; the decontaminated subset (`decontaminate.py`) measures
generalization. Report both.

## Research artifacts (parked, not shipped)

- **Hinglish tokenizer:** 800 Hinglish + 200 Devanagari whole-word tokens save
  **16.57% on Hinglish / 12.62% overall** on held-out text. The 16 GB M4
  cannot calibrate the 1,000 new output rows alongside LoRA; a 24 GB GPU or a
  row-sparse optimizer is needed. Scripts and reports are in the repo.
- **SFT:** two arms were trained and rejected. The injected-token model garbled
  words; the stock-base model applied the crop template to snake bites. The
  negative results are documented in `docs/v4_ab_plan.md` and
  `docs/v4_progress.md`.

## Intended use

- Offline Hinglish/Hindi/English chat on a phone or laptop.
- Scheme guidance that says "I cannot check live data" and points to the
  official portal/CSC.
- Health questions where the system must **not** improvise: it returns a fixed
  protocol or a safe referral.
- A voice-output loop for low-literacy users.

## Limitations

- **Not a doctor.** The health layer is fixed text, not a diagnosis. It is not
  a substitute for a clinician, and the protocols have not yet been reviewed by
  one.
- **Routing is keyword-based.** Critical-intent recall was 62/63 (98.4%) on the
  first run of an unseen paraphrase set; a novel phrasing can still fall
  through to the model. Do not quote the 40-prompt 100% as a population
  estimate.
- **Not a live-data source.** Mandi prices, weather, exam dates, train status
  and scheme installments are not available; the system says so.
- **2B parameters.** Non-health open-ended answers can be generic or wrong.
  RAG is the intended fix for facts, not memorization.
- **Hinglish-first.** Devanagari and English are supported, but the base model
  sometimes answers Hinglish in Devanagari.
- **Benchmark scope.** 200 prompts, five domains; a usability benchmark, not a
  leaderboard.

## License

Apache-2.0, following Qwen3.5-2B. The router, safety protocols, RAG index and
benchmark are original work under the same license.

## Citation

```bibtex
@misc{bharat_tiny_llm_v4,
  title  = {Bharat-Tiny-LLM v4: A Safety-First Offline Hinglish System},
  author = {Kishore, Gautam},
  year   = {2026},
  note   = {eulogik. Apache-2.0. Stock Qwen3.5-2B + local safety/router/RAG.},
  url    = {https://github.com/eulogik/Bharat-Tiny-LLM}
}
```
