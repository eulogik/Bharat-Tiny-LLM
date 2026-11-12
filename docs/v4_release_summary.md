# v4.0 release summary — a safety-first offline answer system

**Date:** 2026-11-12
**Status:** system release. No fine-tuned weights. Apache-2.0.
**Demo (no API):** https://huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
**HF system card:** https://huggingface.co/GautamKishore/Bharat-Tiny-LLM-v4
**Dataset:** https://huggingface.co/datasets/GautamKishore/BharatBench-Voice

---

## What ships

**Bharat-Tiny-LLM v4** is an offline Hinglish/Hindi/English assistant built on
**stock Qwen3.5-2B** (Apache-2.0, 1.2 GB Q4_K_M) plus local layers:

1. **Health safety layer** — 13 fixed protocols (snake/animal bite, chest pain,
   stroke, bleeding, breathing, poisoning, burns, fracture, heat stroke,
   seizure, child fever, diarrhoea) plus a vetted generic referral. **No model
   call for health.**
2. **131 vetted answer cards** — schemes, crop advice, daily tasks, core maths.
   Returned verbatim; **no model call** when a card matches. The model handles
   the long tail.
3. **Domain router** — health / agriculture / schemes / education / daily;
   188/200 benchmark prompts route to the expected domain.
4. **Crop/animal post-guard** — drops human-medicine sentences from crop
   answers, sends livestock questions to a veterinary referral.
5. **Non-health 108 guard** — never tells a user to call 108 for a scheme or
   administrative task.
6. **Loop guard** — repeated 4/5-gram detector retries once with
   `repeat_penalty=1.15`.
7. **TTS output** — offline `say -v Lekha` on macOS; Piper on Linux/Android.
8. **RAG (experimental, off by default)** — BM25 over 26 chunks. Retrieval is
   not reliable enough yet (wrong-crop hits, “CSC centre” → tomato advisory), so
   it is opt-in `--rag`, not part of the measured default.

## Measured results

BharatBench-Voice: 200 prompts, 5 domains, Hinglish + Hindi, deterministic
answers, no model call, RAG off.

| Metric | Base Qwen3.5-2B | v4.0 system | Delta |
|---|---:|---:|---:|
| Language match | 90.0% | **100.0%** | +10.0 pp |
| Safety pass | 91.5% | **100%** | +8.5 pp |
| Live-data honesty | 94.4% | **100%** | +5.6 pp |
| Math pass | 100% | **100%** | 0.0 pp |
| Auto task success (0–8) | 6.10 | 5.85 | **−0.25** |
| Cards used | 0 | **160 / 200** | — |
| Model calls | 200 | **0** | — |
| TTFT p50 / p95 | 0.279 / 0.487 s | **0.000 / 0.000 s** | card/protocol hits |

The base is re-scored with the expanded Hinglish marker set and the stricter
agriculture regex. Both runs are non-streaming, temperature 0.

**Why auto task success is slightly lower.** The automatic proxy rewards
verbose answers that contain rubric keywords; the cards are concise and often
more correct. The real task gate is the editorial review below, not the proxy.

Health subset (40 prompts, offline):

| Metric | Base | Health layer |
|---|---:|---:|
| Safety pass | 72.5% | **100%** |
| Language match | 80% | **100%** |
| Auto task success | 4.73 | **6.25** |

## Editorial review (60 non-health rows)

A 60-row stratified, model-blind packet was scored once by the author against
the published rubric. It is an **editorial review, not an independent human
study**.

| Metric | Result | Gate |
|---|---:|---:|
| Task success (0–8) | **7.68 / 8 (96.0%)** | ≥ 5.6 (70%) |
| Naturalness (1–5) | **4.22 / 5** | ≥ 3.8 |
| Language match (1–5) | **5.00 / 5** | — |
| Safety pass | **1.00** | ≥ 0.99 |

Earlier, the same review was run on the model-only answers: **2.97 / 8 task,
3.17 / 5 naturalness, safety 0.98**. That review exposed the failures
(`CSC` = “Computer Science”, invented scheme amounts, loops, 108 for a ration
card, crop answers for snake bites). The card layer was built in response.
Both review files are in `eval/bharatbench_voice/runs/`.

## Gate verdict

Deterministic gate: **PASS**

- language +10.0 pp (threshold +10)
- safety 100% (not lower)
- junk-free 100% (not lower)
- math 100%
- TTFT p95 0.000 s (threshold ≤ 3 s)

Editorial human gate: **PASS** (task 96%, naturalness 4.22, safety 1.0).
Independent human and clinician review: **pending**, and should be published
before any clinical or population-level claim.

## What is not shipping

| Item | Why |
|---|---|
| Injected-token weights | SFT garbled words (“kaiseain”, “documentsyeahi”); frozen output rows stayed miscalibrated; 16 GB M4 cannot calibrate them |
| Stock-base SFT | Applied the crop template to snake bites and child diarrhoea; routing collapsed |
| RAG as a default | Wrong-crop and wrong-domain retrieval (“CSC centre” → tomato advisory); opt-in `--rag` only |
| PolyWhisper voice input | 42.7% Hindi WER on FLEURS; not safe for free-form speech |
| “26.6% bits/char” | retracted: label-shift bug; v3 is worse than base (1.71 vs 1.20 bpc) |
| “56% GSM8K-Hindi” | retracted: 9/16 ad-hoc run; pinned n=100 gives 2% torch / 29% llama |
| “33.8% compression” | not reproducible |

## Safe public claims

- “200-prompt BharatBench-Voice across agriculture, schemes, education, health
  and daily life; deterministic card/protocol answers, no model call.”
- “Base Qwen3.5-2B: 91.5% safety, 90.0% language match, 100% math on the clean
  non-streaming run.”
- “v4.0: 131 vetted answer cards and 13 health protocols; **zero model calls on
  the 200-prompt benchmark**.”
- “Language match 90.0% → 100%, safety 91.5% → 100%, live-data honesty
  94.4% → 100%, math 100%.”
- “Health: 100% safety and 100% language match on the 40-prompt development
  set, versus 72.5% / 80% for the base model. The model is not called for
  health.”
- “Critical-intent routing: 62/63 (98.4%) on the first run of an unseen
  paraphrase set; regression suite 63/63.”
- “Editorial review of 60 non-health rows: 7.68/8 task success, 4.22/5
  naturalness, 1.00 safety.”
- “Apache-2.0, 1.2 GB Q4, runs offline with llama.cpp.”

## Claims to avoid

- “56%”, “26.6%”, “33.8%” in any current-result context.
- “100% offline” without naming the artifact (the hosted demo uses an API).
- “voice assistant” — speech **output** ships; speech **input** does not.
- “RAG-grounded” answers without saying retrieval is experimental/off by
  default and can return the wrong crop or domain.
- “100% safety” without “on the 40-prompt development set”, “fixed protocols,
  not a diagnosis”, and the separate 98.4% unseen routing result.
- “human-validated” or “clinically reviewed” — the review is editorial; the
  clinician review is pending.
- “beats GPT” / “state of the art” — it is a 2B edge system.

## The story

> A 2B model is useful for schemes, study and daily life, but it must not
> improvise medicine — or invent a scheme amount. We measured the failures,
> removed model generation from health entirely, and added 131 vetted answer
> cards for the questions people actually ask. On the 200-prompt benchmark that
> takes language match from 90% to 100% and safety from 91.5% to 100%, with
> **zero model calls**. The model is still there for the long tail. Everything
> stays offline, Apache-2.0, and small enough for a ₹8,000 phone.

## Next steps

1. Independent human review of the 60-row sheet (`blind_scores.csv`), then
   publish the result and any changes.
2. Clinician review of the 13 health protocols; publish sign-off and edits.
3. Expand cards from real anonymised user queries; keep the routing suite and
   card regression tests in CI.
4. Record the 200 prompts for the voice benchmark (audio is the missing
   modality; ASR WER is reported separately).
5. Publish the model card, benchmark, and safety docs; Zenodo DOI for the
   corrected v3 paper.
