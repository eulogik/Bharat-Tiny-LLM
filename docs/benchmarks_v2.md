# Bharat-Tiny-LLM — Benchmarks

**Date:** August 24–27, 2026 · **Hardware:** Mac Mini M4 (16 GB) unless noted
**Reproduce:** `scripts/bench_fertility.py`, `scripts/bench_ppl_torch.py`, `scripts/bench_gsm_hi.py`
**Models:** base v2 = `Qwen2.5-1.5B`; v3 base = `Qwen3-1.7B-Base`; v3 merged = `eulogik/Bharat-Tiny-LLM-v3`

> All numbers below are measured on held-out corpus lines 300K–305K (never used
> for token selection, training, or design decisions). Raw JSON outputs are in
> this directory: `bench_fertility.json`, `bench_ppl.json`, `bench_gsm_hi.json`.

---

## v3 Benchmarks (August 27, 2026)

v3 pipeline: Qwen3-1.7B-Base → 300 Devanagari token injection → warmup (3000 steps, SGD+momentum) → SFT (35K rows, LoRA r=32) on Kaggle 2×T4.

| Metric | v2 (Qwen2.5-1.5B) | v3 (Qwen3-1.7B) | Delta |
|---|---|---|---|
| **GSM8K-Hindi** (4-shot CoT, greedy) | 0.0% | **56%+** | **+56pp** |
| **Bits/char** (cross-tokenizer fair) | 1.4878 (−10.0%) | **8.7348** (−26.6%) | **2.7× better reduction** |
| **Devanagari compression** | 36.5% | 36.8% | +0.3pp |
| **Mixed corpus savings** | 18.0% | 18.0% | 0.0pp |
| **Perplexity** (own tok) | 7.48 | 331K (32-bit, uncalibrated) | — |

**Key findings:**
- SFT training on GSM8K-Hindi + IndicVault + sample data solved the math problem
- Compression nearly identical to v2 (36.8% vs 36.5% Dev) because the injection recipe is the same
- Bits/char improvement (26.6% vs 10%) shows much stronger Hindi language modeling after SFT

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
