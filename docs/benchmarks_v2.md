# Bharat-Tiny-LLM — Benchmarks

> **⚠️ Correction (2026-11-11).** The v3 bits/char result below was produced by
> `scripts/bench_ppl_torch.py`, which shifted labels twice and therefore scored
> token *i+2* instead of *i+1*. The `11.90 → 8.73 (−26.6%)` row and the `331K`
> perplexity are invalid. The corrected benchmark (`scripts/bench_bpc.py`) gives
> **1.20 bpc (base) vs 1.71 bpc (v3)** on a 60-window smoke run, and an
> independent llama.cpp/Q4 run gives **1.51 vs 1.72** — v3 is worse in
> bits/char. The 56% GSM8K-Hindi row is also retracted; pinned n=100 protocols
> give 2% (torch/fp16) and 29% (llama.cpp/Q4). See `docs/claim_audit.md`.
> Tokenizer fertility (36.9% Devanagari, 17.7% mixed) remains valid.

**Date:** August 24–27, 2026 · **Hardware:** Mac Mini M4 (16 GB) unless noted
**Reproduce:** `scripts/bench_fertility.py`, `scripts/bench_bpc.py`, `scripts/bench_gsm_hi.py`
**Models:** base v2 = `Qwen2.5-1.5B`; v3 base = `Qwen3-1.7B-Base`; v3 merged = `eulogik/Bharat-Tiny-LLM-v3`

> All numbers below are measured on held-out corpus lines 300K–305K (never used
> for token selection, training, or design decisions). Raw JSON outputs are in
> this directory: `bench_fertility.json`, `bench_ppl.json`, `bench_gsm_hi.json`.

---

## v3 Benchmarks (August 27, 2026)

v3 pipeline: Qwen3-1.7B-Base → 300 Devanagari token injection → warmup (3000 steps, SGD+momentum) → SFT (35K rows, LoRA r=32) on Kaggle 2×T4.

| Metric | v2 (Qwen2.5-1.5B) | v3 (Qwen3-1.7B) | Delta |
|---|---|---|---|
| **GSM8K-Hindi** (4-shot CoT, greedy, llama.cpp/Q4, n=100) | not measured | **29%** (v3.3: 35%) | — |
| **GSM8K-Hindi** (same prompts, torch/fp16, n=100) | not measured | 2% | stack-sensitive |
| **Bits/char** (corrected, 60-window smoke) | 1.49 vs 1.65 | **1.71 vs 1.20** | **v3 worse** |
| **Devanagari compression** (tokenizer only) | 36.5% | 36.8% | +0.3pp |
| **Mixed corpus savings** (tokenizer only) | 18.0% | 18.0% | 0.0pp |
| **Perplexity** (own tok) | 7.48 | 11.09 (corrected) | — |

**Key findings:**
- SFT did **not** create a math breakthrough. The 56% figure was a 9/16 ad-hoc
  run and is retracted; pinned n=100 gives 29% under llama.cpp/Q4 and 2% under
  torch/fp16 on identical prompts and weights.
- Compression is nearly identical to v2 (36.8% vs 36.5% Dev) because the
  injection recipe is the same. It is a tokenizer result, not a quality result.
- The corrected bits/char shows v3 is **worse** than base (1.71 vs 1.20). The
  old "26.6% better" was a label-shift bug.

### GGUF files for edge deployment

| File | Size | Quantization |
|---|---|---|
| `bharat-tiny-llm-v3-f16.gguf` | 3.2 GB | F16 (reference) |
| `bharat-tiny-llm-v3-q4_k_m.gguf` | 1.0 GB | Q4_K_M (edge) |

Both available at `eulogik/Bharat-Tiny-LLM-v3` on HuggingFace Hub.

---

## v2 Benchmarks (August 24, 2026)

## 1. Token compression (fertility) — injection only, no training involved

| Metric | Base tokenizer | +300 Brahmi tokens | Savings |
|---|---|---|---|
| Tokens / 1000 chars — **Devanagari runs** | 1041.9 | 661.6 | **36.5%** |
| Chars / token — Devanagari runs | 0.960 | 1.511 | +57.4% |
| Tokens / 1000 chars — **full mixed conversations** | 593.3 | 486.6 | **18.0%** |

*Held-out: 5,000 conversations, 983K Devanagari chars (243,887 runs), 2.75M total chars.*

**Disclosure:** end-to-end savings (18%) are lower than script-only savings (36.5%)
because Latin/Hinglish spans gain nothing. Products serving mostly Hinglish will
see proportionally smaller wins.

### Example phrases (measured, same held-out protocol)

| Phrase | Base | v2 | Δ |
|---|---:|---:|---|
| ज़रूरी बात है क्या करते हो | 25 | 11 | −56.0% |
| हम आपको बताना चाहते हैं | 23 | 16 | −30.4% |
| नमस्ते, आप कैसे हैं? | 20 | 13 | −35.0% |
| कितने बजे मिलना है? | 18 | 13 | −27.8% |
| चाय पी लो, फिर बात करते हैं | 25 | 19 | −24.0% |

## 2. Budget sweep (design-time analysis; validation lines 10K–15K)

From `models/budget_analysis.json`:

| Budget (tokens) | Tokens saved % | Incremental vs prev |
|---:|---:|---:|
| 50 | 14.19 | +14.19 |
| 100 | 22.05 | +7.86 |
| 200 | 31.37 | +9.32 |
| **300** | **36.25** | +4.88 |
| 500 | 42.89 | +6.64 |
| 750 | 48.09 | +5.20 |

300 tokens sits past the steepest gains at near-zero warmup cost.

## 3. Likelihood on held-out Hindi/Hinglish (cross-tokenizer fair)

Raw PPL is not comparable across different vocabularies; we compare
character-normalized cross-entropy on the **identical character prefix**
(~161K chars, verified coverage within 0.6%):

| Model | PPL (own tokenizer) | Bits / char ↓ |
|---|---:|---:|
| Qwen2.5-1.5B bf16 | 6.11 | 1.6531 |
| **v2 Q4 + LoRA** | 7.48 | **1.4878** |

**→ 10.0% fewer bits per character.**
(The higher raw PPL is an artifact of predicting 18–26% fewer tokens per char.)

Training-phase context (from training logs, own-validation):
embedding warmup val loss 2.776 → 1.320 (−52.5%); final LoRA val loss 1.837 @ step 400.
LoRA config (MLX run): rank 8, scale 20, dropout 0, 16 layers, LR 1e-4, 500 steps (~35 min).

## 4. GSM8K-Hindi (grade-school math, 4-shot CoT, greedy)

Dataset: `bingbangboom/gsm8k-hindi` test_main, N=100, exact match on final number.
Gold parsing accepts `####N` and `###N` separators (28% of rows use `###`).

| Model | Accuracy | Decode tok/s |
|---|---:|---:|
| Base Qwen2.5-1.5B (bf16) | 3.0% | 27.1 |
| v2 Q4 + LoRA | **0.0%** | **52.7** |

**Honest reading:**
- Neither model can do Hindi grade-school math at this scale (base ≈ floor).
- v2 scores *below* base here: LoRA fine-tuning on conversation data taught it to
  chat instead of following few-shot CoT formats — most answers lack the
  `उत्तर:` marker entirely (`pred=None`). v2 trades instruction-formatting for
  conversational fluency.
- Decode throughput on the identical protocol/hardware: v2 ≈ 1.9× base
  (52.7 vs 27.1 tok/s). Confounds: v2 is Q4-quantized and uses the compressed
  tokenizer, so this is end-to-end product speed, not an architecture claim.
- Raw JSON: `docs/bench_gsm_hi.json` (100 samples/model, prompts + predictions).

**Implication for positioning:** sell v2 as an *offline Hindi/Hinglish
conversational assistant* with superior token economics — not as a reasoning or
math model. Capability work (DPO/instruction data, stronger base) belongs to v3.

## 5. Known product quirks (discovered during release testing)

1. **Wrong adapter was public.** The HF repo's `lora_adapter/adapters.safetensors`
   matched no local checkpoint and produced garbage output. Fixed Aug 24 by
   uploading the verified step-400 checkpoint (sha `422c550b…`). If you downloaded
   before this date, re-download.
2. **No system prompt.** v2 was trained without system-role turns; adding one
   degrades output badly (echo + drift). The `bharat` CLI therefore sends none.
3. **Generation config matters.** temp=0.8 degenerates into garbled CJK/out-of-script
   tokens. Canonical: temp 0.3, top-p 0.85, repetition penalty 1.25, no-repeat-ngram 3.
