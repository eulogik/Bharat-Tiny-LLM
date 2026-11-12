# v4 release summary — what is shipping and what is not

**Date:** 2026-11-12
**Status:** system release. No fine-tuned weights.

---

## What ships

**Bharat-Tiny-LLM v4** is an offline Hinglish/Hindi/English assistant built on
**stock Qwen3.5-2B** (Apache-2.0, 1.2 GB Q4_K_M) plus local system layers:

1. **Domain router** — separates health from agriculture/schemes/education/
   daily. 188/200 benchmark prompts route to the expected domain.
2. **Health safety layer** — fixed, vetted protocols for critical intents and
   a safe referral for everything else. **No model generation for health.**
3. **Crop/animal post-guard** — replaces crop answers that leak human
   medicine/108, and sends livestock questions to a veterinary referral.
4. **Loop guard** — a repeated-5-gram detector retries once with
   `repeat_penalty=1.15`.
5. **RAG (experimental, off by default)** — BM25 over 26 source-linked chunks.
   Retrieval is not reliable enough yet (wrong-crop hits, “CSC centre” →
   tomato advisory), so it is not in the measured default. Enable with
   `--rag` for experiments.
6. **TTS output** — offline `say -v Lekha` on macOS; Piper on Linux/Android.

## Measured results

BharatBench-Voice: 200 prompts, 5 domains, temperature 0,
`enable_thinking=false`, non-streaming, no draft model, RAG off (the default),
health layer on, crop/animal guard and loop retry applied.

| Metric | Base Qwen3.5-2B | v4 system | Delta |
|---|---:|---:|---:|
| Language match | 89.5% | **95.0%** | +5.5 pp |
| Safety pass | 91.5% | **100%** | +8.5 pp |
| Live-data honesty | 94.4% | **100%** | +5.6 pp |
| Math pass | 100% | 100% | 0.0 pp |
| Loop-free (5-gram < 0.50) | 99.5% | **100%** | +0.5 pp |
| Auto task success (0–8) | 6.08 | **6.59** | +0.51 |
| TTFT p50 / p95 | 0.279 / 0.487 s | 0.229 / 0.339 s | faster |
| Total p50 / p95 | 9.10 / 17.47 s | 6.43 / 6.88 s | faster |

The base safety figure is the **re-scored** number with the stricter agriculture
regex (the first run was 93.0%); the candidate includes the crop/animal guard.
Both runs are non-streaming, no draft model, temperature 0.

Health subset (40 prompts, offline):

| Metric | Base | Health layer |
|---|---:|---:|
| Safety pass | 72.5% | **100%** |
| Language match | 80% | **100%** |
| Auto task success | 4.73 | **6.15** |

## What is not shipping

| Item | Why |
|---|---|
| Injected-token weights | SFT garbled words (`kaiseain`, `documentsyeahi`); frozen output rows stayed miscalibrated; 16 GB M4 cannot calibrate them |
| Stock-base SFT | Applied the crop template to snake bites and child diarrhoea; domain routing collapsed |
| RAG as a default | Wrong-crop and wrong-domain retrieval (“CSC centre” → tomato advisory); kept as opt-in `--rag` |
| PolyWhisper voice input | 42.7% Hindi WER on FLEURS; not safe for free-form speech |
| "26.6% bits/char" | retracted: label-shift bug; v3 is worse than base (1.71 vs 1.20 bpc) |
| "56% GSM8K-Hindi" | retracted: 9/16 ad-hoc run; pinned n=100 gives 2% torch / 29% llama |
| "33.8% compression" | not reproducible |

## Safe public claims

- "200-prompt BharatBench-Voice across agriculture, schemes, education, health
  and daily life."
- "Base Qwen3.5-2B: 91.5% safety (re-scored with the stricter crop regex),
  89.5% language match, 100% math on the clean non-streaming run."
- "v4 health layer: **100% safety and 100% language match on the 40-prompt
  health benchmark (used for development)**, versus 72.5% / 80% for the base
  model. The model is not called for health questions."
- "Critical-intent routing: **62/63 (98.4%) on the first run of an unseen
  paraphrase/misspelling/Devanagari set**; after one fix the regression suite
  is 63/63. No critical health prompt in the 200-prompt benchmark is routed to
  the model."
- "Hinglish tokenizer: 16.57% fewer tokens on held-out Hinglish, 12.62%
  overall" — **tokenizer-only, parked, not in the shipped model**.
- "Apache-2.0, 1.2 GB Q4, runs offline with llama.cpp."

## Claims to avoid

- "56%", "26.6%", "33.8%" in any current-result context.
- "100% offline" without naming the artifact (the hosted demo uses an API).
- "voice assistant" — speech **output** ships; speech **input** does not.
- "RAG-grounded" answers without saying retrieval is experimental/off by
  default and can return the wrong crop or domain.
- "100% safety" without "on the 40-prompt development set", "fixed protocols,
  not a diagnosis", and the separate 98.4% unseen routing result.
- "beats GPT" / "state of the art" — it is a 2B edge system.

## Gate verdict

The original v4 gate was "beat the base by ≥10 points on Hinglish task
success". The system does **not** meet that: auto task success moves from 6.08
to **6.59 (+0.51)**, and language match improves by **+5.5 pp**.

What it does meet is the safety requirement:

- Overall safety: 91.5% → **100%**.
- Health safety: 72.5% → **100%**.
- Health language match: 80% → **100%**.
- Live-data honesty: 94.4% → **100%**.
- Overall language match: 89.5% → **95.0%** (+5.5 pp).
- Auto task success: 6.08 → **6.59** (+0.51).
- Critical-intent routing: **62/63 (98.4%)** on the first unseen run; the
  regression suite is 63/63 after one fix.

The honest conclusion is that this is a **safety and health release**, not a
task-success breakthrough. For a health-adjacent assistant, that is the
correct trade: a model that is 10 points better at tasks but tells a snake-bite
victim to band the wound is not shippable. The fine-tuning arms that tried to
buy task success were rejected because they broke routing and coherence.

If the product goal is strictly "highest task success", ship the base model.
If the goal is "usable and safe for rural India", ship the system. The
benchmark makes both choices explicit.

## What the system is not

- Not a new foundation model. The weights are stock Qwen3.5-2B.
- Not a doctor. Health is fixed protocols, not diagnosis.
- Not a voice assistant yet. Speech output ships; speech input does not.
- Not a task-success breakthrough. It is a safety and honesty release.

## The story

> A 2B model is useful for schemes, study and daily life, but it must not
> improvise medicine. We measured the failure, removed model generation from
> health entirely, and published the benchmark. On the 40-prompt development
> set that takes safety from 72.5% to 100% and language match from 80% to 100%.
> Because a safety layer only works if the query reaches it, we also tested
> routing on unseen paraphrases: **62/63 critical intents (98.4%)** on the
> first run, 63/63 after one fix, with no critical health prompt reaching the
> model in the 200-prompt benchmark. The rest of the assistant stays offline,
> Apache-2.0, and small enough for a ₹8,000 phone.

## Next steps

1. Fill the blinded 60-row sheet
   (`eval/bharatbench_voice/runs/blind_scores.csv`) and merge it with
   `score_blind_sheet.py`; gate is task success ≥ 70%, naturalness ≥ 3.8.
   Instructions: [`docs/blind_scoring.md`](blind_scoring.md).
2. Get the safety protocols reviewed by a clinician; publish the review and
   any changes.
3. Expand the routing suite with real anonymised user queries; keep
   `--strict` in CI.
4. Record the 200 prompts for the voice benchmark (audio is the missing
   modality; ASR WER is reported separately).
5. Publish the model card, benchmark, and safety docs; Zenodo DOI for the
   corrected v3 paper.
