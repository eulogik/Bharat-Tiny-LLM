# v4 tokenizer decision — Hinglish injection go/no-go

**Date:** 2026-11-11
**Base:** Qwen3.5-2B (Apache-2.0, 248,320-row padded embedding, 248,077 tokenizer)
**Held-out set:** `data/processed/valid_gold.jsonl` (18,641 conversations, 9.66M chars)
**Mining corpus:** `train_gold_v3.jsonl` (403,649 messages seen)
**Script:** `scripts/mine_hinglish_tokens.py`
**Raw reports:** `data/v4/hinglish_candidates.json`, `data/v4/h1000/hinglish_candidates.json`

---

## Results

| Budget | Hinglish savings | All-text savings | English savings | Devanagari savings |
|---|---:|---:|---:|---:|
| 800 Hinglish + 200 Devanagari **whole words** | **16.57%** | **12.62%** | +0.38% | **10.77%** |
| 800 Hinglish + 200 Devanagari n-grams | 16.56% | 7.41% | +0.21% | 2.24% |
| 1,000 Hinglish + 0 Devanagari | 16.79% | 6.52% | +0.21% | 0.62% |

Token counts on held-out Hinglish: **299.59 → 249.95 tok/1000 chars**. The whole-word Devanagari list is the clear winner: it keeps the
Hinglish gain and lifts full-corpus savings from 7.41% to **12.62%**. The
selected Devanagari tokens are real words (` ज़रूरी`, ` ज़्यादा`, ` हूँ`,
` एकदम`, ` बिलकुल`), not partial subwords.

English is unchanged (244.04 → 243.10; +0.38% savings, i.e. no regression).

## Verdict against the kill criteria

The plan set two thresholds:

1. **Hinglish slice ≥ 12%** → **PASS** at 16.57%.
2. **Overall corpus ≥ 12%** → **PASS** at 12.62% (whole-word Devanagari mix).

This is a clean pass for the 800+200 whole-word list. The earlier n-gram mix
failed the overall threshold (7.41%) because partial Devanagari subwords rarely
match; the word-mode fix is what made the difference.

## What the numbers mean

- On the target slice, the model spends **17% fewer tokens** on the same
  Hinglish text. That is a real latency/cost win on a ₹8,000 phone.
- On mixed and Devanagari text, the gain is small. Qwen3.5's tokenizer already
  covers Devanagari far better than Qwen3's: 637.5 vs 1,043 tok/1000 chars. The
  old v3 injection advantage largely disappears on the newer base.
- The 200 Devanagari n-grams include partial subwords (e.g. `' ्टे'`,
  `' ेकल'`). Round-trip decoding is exact, but these tokens are not words and
  may tokenize inconsistently. If we keep Devanagari, mine **whole words**, not
  arbitrary n-grams.
- English is not harmed. The `max_english_ratio=0.35` filter and the
  `ENGLISH_COMMON` blocklist did their job.

## Decision

**Proceed with the 800 Hinglish + 200 Devanagari whole-word mix for warmup +
SFT, and freeze tokenizer mining here.** The tokenizer is an enabler, not the
product. The next gate is task success and naturalness on BharatBench-Voice,
not another compression point.

If the warmup cannot recover the base model's bits/char, or the SFT fails to
beat the base by 10 points on Hinglish task success, **ship the base model with
prompting + RAG and drop the custom tokenizer from the launch story.** The
Apache-2.0 weights, offline GGUF, voice loop, and benchmark remain.

## Why not a bigger token budget

- 1,000 Hinglish tokens add only 0.23pp on Hinglish; the curve is flat.
- The whole-word Devanagari list is the efficient allocation: 12.62% overall
  with the same 1,000-token budget.
- Each added token costs an embedding row and a longer tail of rare tokens;
  the warmup has to calibrate all of them.
- The 248K Qwen3.5 vocabulary is already large. A 1K addition is +0.4%; beyond
  that, the engineering and eval burden grows faster than the savings.
- The product bottleneck is quality and domain grounding, not vocabulary.

## Open risks

1. **Partial-word tokens.** The Devanagari n-grams may produce odd
   tokenization. Mitigation: whole-word mode, or drop Devanagari tokens.
2. **Warmup cost.** The new rows must be mean-initialized and warmed; a random
   init will look broken for the first hundreds of steps.
3. **Evaluation confound.** Token savings change the denominator of
   bits/char. Always report bits/char and bits/byte, not just tokens.
4. **No quality win yet.** The corrected v3 benchmark shows injection+SFT can
   be *worse* in bits/char. The v4 SFT must be measured against the base on
   the same benchmark; token count alone is not evidence.

## Next actions

1. Inject the 800+200 list into Qwen3.5-2B with mean init
   (`scripts/inject_v4_tokens.py`).
2. Warm up the new rows on Hinglish text
   (`scripts/warmup_v4_embeddings.py`); target: recover base bits/char within
   5% and cut new-token NLL by >50%.
3. Run BharatBench-Voice on base vs v4; gate on ≥10-point task-success win.
4. Only then build the voice/RAG/APK layer.
