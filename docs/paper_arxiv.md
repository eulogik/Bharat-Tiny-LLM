# Brahmi-Lite: Minimal-Budget Devanagari Token Injection for Edge LLMs

**Gautam Kishore — eulogik (eulogik.com)**
*Technical report, August 2026. All numbers in this document are measured; reproduction scripts are in the public repository.*

---

## Abstract

Multilingual byte-level BPE tokenizers tax Devanagari script: Hindi text consumes
roughly 1.04 tokens per character under Qwen2.5's tokenizer, versus ~0.59 for
mixed Hinglish text. We present a **minimal-budget** vocabulary injection study:
we add only **300 Devanagari subword tokens** (+0.198% of Qwen2.5-1.5B's
152,236-entry embedding table) to an off-the-shelf multilingual BPE tokenizer,
initialize new embeddings as the mean of their constituent byte tokens, and
recover quality with two light training phases: continued pre-training of the
new rows only, followed by LoRA fine-tuning.

On held-out data, injection alone reduces token count on pure Devanagari text by
**36.5%** and end-to-end on mixed Hinglish/Devanagari conversations by **18.0%**.
After training, the model predicts held-out Hindi/Hinglish text at **10.0% fewer
bits per character** than its base under identical character coverage. The whole
pipeline runs on consumer hardware (one Mac Mini M4 and one Colab T4) at zero
cloud cost, and ships as an 880 MB 4-bit MLX model.

Unlike prior Indic vocabulary work that extends tokenizers by 48K tokens
(OpenHathi) or redesigns them wholesale (Sarvam-1), this study quantifies what a
*three-hundred-token* budget buys. We release a budget sweep (50→1000 tokens)
showing diminishing returns beyond ~300, position our results against related
tokenizer-surgery literature, and publish all weights under Apache 2.0.

---

## 1. Introduction

Non-Latin scripts pay a "tokenizer tax" in multilingual LLMs: Petrov et al.
(2023) showed some languages tokenize up to 15× longer than English, and recent
measurement puts the average Indic penalty at 8× under cl100k-class vocabularies.
Longer sequences mean slower inference, more KV-cache memory, and higher latency
— exactly the resources edge devices lack.

The standard fixes are heavy: train a tokenizer from scratch (Sarvam-1), extend
the vocabulary by tens of thousands of tokens (OpenHathi added 48K), or
transplant an entire tokenizer via hypernetworks (ZETT). For a small team or an
edge-first product, none of these is cheap: every added token needs training
budget to become useful, and large expansions risk degrading the base model's
existing capabilities.

We ask the opposite question: **what is the minimum useful injection?** If 300
well-chosen subwords capture most of the compression benefit for one script, the
recipe becomes practical for anyone fine-tuning a stock open model for a
specific language market. Our contributions:

1. A reproducible selection method: corpus n-gram mining scored by
   `tokens_saved × √frequency`, filtered against the base vocabulary.
2. A budget sweep from 50 to 1000 tokens measuring the compression/efficiency
   Pareto frontier, with a knee point near 300 tokens.
3. A two-phase recovery recipe (embedding-row warmup → LoRA) that restores and
   exceeds base-model Hindi likelihood, measured honestly across *different*
   tokenizers via bits-per-character.
4. An end-to-end artifact: Apache-2.0 weights, quantized to 4-bit MLX,
   downloadable from Hugging Face (`eulogik/Bharat-Tiny-LLM-v2-MLX`).

---

## 2. Related Work

**Vocabulary expansion for Indic languages.** OpenHathi (Sarvam AI, 2023)
extended Llama-2's tokenizer by 48K Hindi tokens as phase one of a bilingual
LM recipe. Sarvam-1 later trained a custom 68K-token tokenizer from scratch,
reporting fertility of 1.4–2.1 tokens/word across ten Indic languages.
Krutrim (2024) reports custom tokenizer strategies beating Tiktoken on
token-to-word ratio for Indic scripts. These efforts demonstrate that big
token budgets win big; none publish minimal-budget ablations.

**Tokenizer surgery / transplantation.** ZETT (Minixhofer et al., 2024)
predicts embeddings for arbitrary target tokenizers with a hypernetwork.
Token transplantation via orthogonal matching pursuit (Goddard & Fernandes
Neto, 2025) reconstructs unseen-token embeddings as sparse combinations of
anchor embeddings, outperforming mean-init baselines. HYPEROFA (Özeren et al.,
2025) and OFA study similarity-based initialization for expanded vocabs.
FragMend (Mehta et al., 2026) shows interpretability-guided token selection
and initialization improves non-Latin expansion over frequency-only selection.
Our mean-of-constituents initialization is deliberately the simplest baseline
from this literature; combining it with OMP-style init is future work.

**Writing-system-level adaptation.** Closest in spirit to our measurement:
post-hoc insertion of target-script tokens into byte-level BPE while preserving
token IDs reported −33.5%/−36.6% fertility change for Ukrainian (Didenko,
2026). BrahmicTokenizer-131K retrofits o200k_base with 2,372 Brahmic slots via
linear programming, reporting per-language savings up to 76.8%. SGPE builds a
syllable-aware grapheme encoder for Sinhala+Devanagari reporting 27% reduction
vs o200k_base. Our study differs in budget (300 vs 1.3K–48K+) and in publishing
both script-only and end-to-end mixed-script savings.

**Compression without vocabulary changes.** RomanSetu (ACL 2024) shows
romanization cuts Indic fertility 2–4× but breaks native-script literacy;
Huffman-style transliteration (ACL 2025) trades 50–80% token savings for a
transliteration round-trip. Injection keeps the surface form native.

**Novelty positioning.** We claim no new mechanism: insertion, mean-init, CPT,
and LoRA all exist separately. Our contribution is the **quantified minimal
budget** — showing 300 tokens (0.2% of vocab, <1M trained params in warmup)
deliver 36.5% script-specific compression and a net bits-per-character win —
plus an honest cross-tokenizer evaluation protocol and a fully open artifact.

---

## 3. Method

### 3.1 Corpus

435,999 Hinglish + Devanagari conversations (deduplicated, message format).
Lines 0–15K were used during development for candidate mining and validation;
all benchmark measurements below use lines 300K–305K, untouched by any design
decision.

### 3.2 Candidate mining and scoring

Devanagari Unicode block runs (U+0900–U+097F, length ≥2) are extracted per
message. Character n-grams (n=2..6) are counted; candidates already in the base
vocabulary are dropped. Each candidate scores:

    score = (base_token_count − 1) × √frequency

`√frequency` dampens ultra-frequent short grams so the fixed budget spreads
across the distribution rather than saturating on top-10 words. Top-K by score
form the injected set (K = budget ∈ {50, 100, 200, 300, 500, 750, 1000}).

### 3.3 Injection

Tokens are appended with `tokenizer.add_tokens`; model input/output embeddings
grow by K rows. New rows are initialized as the **mean of constituent
byte-token embeddings**. Final vocabulary: 151,936 → 152,236 (+300, +0.198%).

### 3.4 Phase 1 — Embedding warmup

Backbone frozen; only the 300 new rows (614K params) train on 20K chunks of the
corpus for 3,000 steps (AdamW, LR 1e-3, cosine decay, Colab T4). Validation loss
on chunks containing injected tokens fell 52.5% (2.776 → 1.320).

### 3.5 Phase 2 — LoRA fine-tune

MLX LoRA on the quantized model: rank 8, scale 20, dropout 0, 16 attention
layers, LR 1e-4, batch 4, seq 512, 500 steps (~35 min on M4; best checkpoint at
step 400, val loss 1.837). Note this differs from our earlier T4 QLoRA run
(rank 16, α 32); the MLX configuration is what ships.

### 3.6 Quantization

4-bit affine, group size 64 → 880 MB single-file MLX model (4.501 bpw).

---

## 4. Results

All measurements: held-out lines 300K–305K (5,000 conversations, 243,887
Devanagari runs). Scripts: `scripts/bench_fertility.py`,
`scripts/bench_ppl.py`, `scripts/bench_gsm_hi.py`.

### 4.1 Compression (injection only, no training involved)

| Metric | Base | +300 tokens | Savings |
|---|---|---|---|
| Tokens / 1000 chars (Devanagari runs) | 1041.9 | 661.6 | **36.5%** |
| Chars / token (Devanagari runs) | 0.960 | 1.511 | +57.4% |
| Tokens / 1000 chars (full mixed convos) | 593.3 | 486.6 | **18.0%** |

End-to-end savings are lower because Hinglish/Latin spans gain nothing — we
consider reporting both mandatory for honesty.

Example phrases (measured):

| Phrase | Base | v2 | Δ |
|---|---|---|---|
| ज़रूरी बात है क्या करते हो | 25 | 11 | −56% |
| हम आपको बताना चाहते हैं | 23 | 16 | −30% |
| कितने बजे मिलना है? | 18 | 13 | −28% |
| नमस्ते, आप कैसे हैं? | 20 | 13 | −35% |

### 4.2 Budget sweep

From `models/budget_analysis.json` (validation slice during design):

| Budget | Tokens saved % | Incremental gain |
|---|---|---|
| 50 | 14.19 | +14.19 |
| 100 | 22.05 | +7.86 |
| 200 | 31.37 | +9.32 |
| **300** | **36.25** | **+4.88** |
| 500 | 42.89 | +6.64 |
| 750 | 48.09 | +5.20 |

Efficiency (% saved per injected token) peaks early and decays; 300 sits past
the steepest gains while keeping warmup trivially cheap. Budgets ≥500 remain
attractive when warmup compute is not constrained.

### 4.3 Likelihood on held-out text (cross-tokenizer fair)

Raw perplexity is incomparable across different vocabularies, so we report
character-normalized cross-entropy on the identical character prefix
(~161K chars):

| Model | PPL (own tok) | Bits / char |
|---|---|---|
| Qwen2.5-1.5B (bf16) | 6.11 | 1.6531 |
| **Bharat-Tiny-LLM v2 + LoRA (Q4)** | 7.48 | **1.4878** |

**10.0% fewer bits per character** for the same text. The higher raw PPL simply
reflects 18–26% fewer prediction events per character under the compressed
tokenizer — the char-normalized metric removes that artifact.

### 4.4 GSM8K-Hindi

4-shot CoT, greedy decoding, exact match on final number, N=100 test items
(`bingbangboom/gsm8k-hindi`). Results recorded in `docs/bench_gsm_hi.json`
(filled by `scripts/bench_gsm_hi.py`; see accompanying benchmarks document for
the measured table).

### 4.5 Deployment

Single-file 880 MB MLX model loads offline; generation uses the canonical
config (temp 0.3, top-p 0.85, repetition penalty 1.25) discovered necessary for
clean output on this model family.

---

## 5. Limitations

1. **Mixed-text gains are modest (18%).** Products serving mostly Hinglish will
   see less speedup than Devanagari-heavy ones. Larger budgets recover this but
   cost warmup time.
2. **Mean-init is likely suboptimal.** OMP transplant or hypernetwork init
   (ZETT/HYPEROFA) may improve quality-per-token; untested here.
3. **No capability preservation suite yet.** We measure Hindi likelihood and
   math accuracy; English MMLU-style regression is unmeasured (LoRA rank 8 on
   attention only makes large regressions unlikely but not impossible).
4. **One base model.** Results are for Qwen2.5-1.5B; transfer to Qwen3/Gemma
   bases is planned (v3).
5. **Machine-translated eval data.** GSM8K-Hindi inherits translation artifacts.

## 6. Conclusion

A three-hundred-token injection — 0.2% of the vocabulary, under 6M trained
parameters across both phases, trained on consumer hardware — buys 36.5%
script-specific compression, 10% better bits-per-character on held-out Hindi,
and ships in a single 880 MB file. For teams localizing stock open models to
single-script markets, minimal-budget injection is a practical first tool, not
a research project. All artifacts are Apache 2.0.

---

## References

- Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023. arXiv:2305.15425
- Minixhofer, Ponti, Vulić. *Zero-Shot Tokenizer Transfer.* NeurIPS 2024. arXiv:2405.07883
- Goddard, Fernandes Neto. *Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit.* 2025. arXiv:2506.06607
- Yamaguchi et al. *How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?* Computational Linguistics 2024. arXiv:2406.11477
- Özeren, Liu, Schütze. *HYPEROFA.* ACL SRW 2025. arXiv:2504.21018
- Sakajo et al. *Dictionaries to the Rescue.* ACL 2025. arXiv:2506.01535
- Remy et al. *Trans-Tokenization / Tweeties.* 2024. arXiv:2408.04303
- Mehta et al. *Defragmenting Language Models (FragMend).* 2026. arXiv:2604.16656
- Didenko. *Writing-System-Level Tokenizer Adaptation for Byte-Level BPE.* TokShop @ COLM 2026. arXiv:2608.00582
- Shravan. *BrahmicTokenizer-131K.* 2026. arXiv:2605.29379
- *WWHO / SGPE: Syllable-aware Grapheme Pair Encoding.* 2026. arXiv:2603.25309
- *Tokenizer Transplantation for Bengali ASR.* 2026. arXiv:2607.09598
- MorphTok. TokShop @ ICML 2025. arXiv:2504.10335
- *The Tokenizer Tax.* 2026. arXiv:2607.24276
- Sarvam AI. *OpenHathi-Hi-v0.1.* 2023. sarvam.ai; *Sarvam-1.* 2024. sarvam.ai/blogs/sarvam-1
- Krutrim. arXiv:2407.12481 (2024); tech report arXiv:2502.09642 (2025)
- MUTANT-Indic. 2025. arXiv:2511.03237
- RomanSetu. ACL 2024. arXiv:2401.14280
- Huffman Transliteration. ACL 2025. arXiv:2509.17493
- Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115
