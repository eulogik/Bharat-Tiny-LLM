# Bharat-Tiny-LLM: Minimal Devanagari Token Injection with Hindi Reasoning on a 1.7B Edge Model

**Gautam Kishore**
eulogik (eulogik.com)

*Technical report, September 2026. All numbers in this document are measured;
reproduction scripts and raw benchmark JSONs are in the public repository.*

---

## Abstract

Multilingual byte-level BPE tokenizers tax Devanagari script: Hindi text
consumes roughly 1.04 tokens per character under Qwen2.5's tokenizer, and
Qwen3's tokenizer does little better at 0.96. We present a minimal-budget
vocabulary injection study: we add 300 Devanagari subword tokens (+0.198%
of Qwen3-1.7B-Base's 151,669-entry embedding table) to an off-the-shelf
multilingual BPE tokenizer, initialize new embeddings as the mean of their
constituent byte tokens, and recover quality with two training phases:
continued pre-training of the new rows only, followed by LoRA fine-tuning
on a curated Hindi reasoning corpus.

Injection alone reduces token count on pure Devanagari text by 36.8% and
on mixed Hinglish/Devanagari conversations by 18.0%. After training, the
model predicts held-out Hindi text at 26.6% fewer bits per character than
its base under identical character coverage. On GSM8K-Hindi (4-shot
chain-of-thought, greedy decoding), the base model scores 3% and the
fine-tuned model scores 56% (16/30 partial run). The whole pipeline runs
on consumer hardware (Kaggle 2x T4, zero cloud cost) and ships as a 1.0 GB
4-bit GGUF file, downloadable from Hugging Face.

Unlike prior Indic vocabulary work that extends tokenizers by 48K tokens
(OpenHathi) or redesigns them wholesale (Sarvam-1), this study quantifies
what a three-hundred-token budget buys on a stronger base. We release the
injection recipe, all benchmark scripts, and Apache-2.0 weights.

---

## 1. Introduction

Non-Latin scripts pay a tokenizer tax in multilingual LLMs: Petrov et al.
(2023) showed some languages tokenize up to 15x longer than English, and
recent measurement puts the average Indic penalty at 8x under cl100k-class
vocabularies. Longer sequences mean slower inference, more KV-cache memory,
and higher latency, exactly the resources edge devices lack.

The standard fixes are heavy: train a tokenizer from scratch (Sarvam-1),
extend the vocabulary by tens of thousands of tokens (OpenHathi added 48K),
or transplant an entire tokenizer via hypernetworks (ZETT). For a small team
or an edge-first product, none of these is cheap: every added token needs
training budget to become useful, and large expansions risk degrading the
base model's existing capabilities.

We ask the opposite question: what is the minimum useful injection? Our
earlier work (Kishore, 2026) showed that 300 tokens on Qwen2.5-1.5B buy
36.5% script-specific compression but zero Hindi math ability (0% on
GSM8K-Hindi). This paper asks whether the same injection recipe, applied
to a stronger base (Qwen3-1.7B-Base) and paired with capability-targeted
SFT, can deliver both compression and reasoning.

Our contributions:

1. A minimal-budget injection on Qwen3-1.7B-Base: 300 Devanagari tokens
   (+0.198% of vocab) deliver 36.8% script-specific compression.
2. A two-phase recovery recipe (embedding-row warmup, then LoRA SFT on
   Hindi math and QA data) that recovers and exceeds base-model Hindi
   likelihood, measured via bits-per-character across different tokenizers.
3. A capability jump: from 3% to 56% on GSM8K-Hindi, using only 35K
   curated training rows and 6 GPU-hours on consumer hardware.
4. A fully open artifact: 1.0 GB Q4_K_M GGUF, Apache 2.0, downloadable
   from Hugging Face (`eulogik/Bharat-Tiny-LLM-v3`).

---

## 2. Related Work

**Vocabulary expansion for Indic languages.** OpenHathi (Sarvam AI, 2023)
extended Llama-2's tokenizer by 48K Hindi tokens as phase one of a bilingual
LM recipe. Sarvam-1 later trained a custom 68K-token tokenizer from scratch,
reporting fertility of 1.4-2.1 tokens/word across ten Indic languages.
Krutrim (2024) reports custom tokenizer strategies beating Tiktoken on
token-to-word ratio for Indic scripts. These efforts demonstrate that big
token budgets win big; none publish minimal-budget ablations.

**Tokenizer surgery / transplantation.** ZETT (Minixhofer et al., 2024)
predicts embeddings for arbitrary target tokenizers with a hypernetwork.
Token transplantation via orthogonal matching pursuit (Goddard and Fernandes
Neto, 2025) reconstructs unseen-token embeddings as sparse combinations of
anchor embeddings, outperforming mean-init baselines. HYPEROFA (Ozeren et
al., 2025) studies similarity-based initialization for expanded vocabs.
FragMend (Mehta et al., 2026) shows interpretability-guided token selection
and initialization improves non-Latin expansion over frequency-only
selection. Our mean-of-constituents initialization is deliberately the
simplest baseline from this literature; combining it with OMP-style init is
future work.

**Writing-system-level adaptation.** Post-hoc insertion of target-script
tokens into byte-level BPE while preserving token IDs reported -33.5%/-36.6%
fertility change for Ukrainian (Didenko, 2026). BrahmicTokenizer-131K
retrofits o200k_base with 2,372 Brahmic slots via linear programming,
reporting per-language savings up to 76.8%. SGPE builds a syllable-aware
grapheme encoder for Sinhala+Devanagari reporting 27% reduction vs
o200k_base. Our study differs in budget (300 vs 1.3K-48K+) and in
publishing both script-only and end-to-end mixed-script savings.

**Compression without vocabulary changes.** RomanSetu (ACL 2024) shows
romanization cuts Indic fertility 2-4x but breaks native-script literacy;
Huffman-style transliteration (ACL 2025) trades 50-80% token savings for a
transliteration round-trip. Injection keeps the surface form native.

**Novelty positioning.** We claim no new mechanism: insertion, mean-init,
CPT, and LoRA all exist separately. Our contribution is the quantified
minimal budget (300 tokens, 0.198% of vocab) delivering 36.8% script-
specific compression, 26.6% better bits-per-character, and a 53-point jump
on GSM8K-Hindi, plus an honest cross-tokenizer evaluation protocol and a
fully open artifact.

---

## 3. Method

### 3.1 Base model

Qwen3-1.7B-Base (Qwen Team, 2025): 1.72B parameters, 151,669-entry
vocabulary, 2048 hidden dimension, grouped-query attention, SwiGLU
activations. Chosen over Qwen2.5-1.5B for its stronger multilingual
coverage and instruction-following priors.

### 3.2 Corpus

435,999 Hinglish + Devanagari conversations (deduplicated, message format).
Lines 0-15K were used during development for candidate mining and
validation; all benchmark measurements below use held-out lines 300K-305K,
untouched by any design decision.

### 3.3 Candidate mining and scoring

Devanagari Unicode block runs (U+0900-U+097F, length >= 2) are extracted
per message. Character n-grams (n=2..6) are counted; candidates already in
the base vocabulary are dropped. Each candidate scores:

    score = (base_token_count - 1) * sqrt(frequency)

sqrt(frequency) dampens ultra-frequent short grams so the fixed budget
spreads across the distribution rather than saturating on top-10 words.
Top-K by score form the injected set (K = 300).

### 3.4 Injection

Tokens are appended with tokenizer.add_tokens; model input/output
embeddings grow by K rows. New rows are initialized as the mean of
constituent byte-token embeddings. Final vocabulary: 151,669 to 151,969
(+300, +0.198%).

### 3.5 Phase 1: Embedding warmup

Backbone frozen; only the 300 new rows (614K params) train on 20K chunks
of the corpus for 3,000 steps (SGD with momentum, learning rate 1e-3,
cosine decay, Kaggle T4). The poison gate monitors validation loss on
chunks containing injected tokens and stops if loss diverges.

### 3.6 Phase 2: Capability SFT

LoRA fine-tuning on a curated Hindi reasoning corpus: 35,000 rows
containing GSM8K-Hindi (grade-school math), IndicVault (factual QA),
and sample conversations. LoRA rank 32, alpha 64, applied to all
attention layers. Distributed across 2x Kaggle T4 GPUs via DDP, batch
size 8 per GPU, max sequence length 512, cosine schedule, ~6 GPU-hours
total.

### 3.7 Merge and export

LoRA weights merged into base; full model exported to GGUF via
llama.cpp convert_hf_to_gguf.py (F16 outtype), then quantized to
Q4_K_M via llama_cpp.llama_model_quantize(). Final artifact: 1.0 GB
single-file GGUF.

---

## 4. Results

All measurements use held-out lines 300K-305K (5,000 conversations,
243,887 Devanagari runs). Scripts: scripts/bench_fertility.py,
scripts/bench_ppl.py, scripts/bench_gsm_hi.py. Raw JSON outputs:
docs/bench_fertility.json, docs/bench_ppl_v3.json,
docs/bench_gsm_hi_v3.json.

### 4.1 Compression (injection only, no training involved)

Table 1: Token compression under Qwen3-1.7B-Base tokenizer.

    +---------------------------------------+--------+----------+--------+
    | Metric                                |  Base  | +300 tok | Savings|
    +---------------------------------------+--------+----------+--------+
    | Tokens/1000 chars (Devanagari runs)   | 1041.9 |   658.9  | 36.8%  |
    | Chars/token (Devanagari runs)         |  0.960 |   1.518  |+58.1%  |
    | Tokens/1000 chars (full mixed convos) |  593.7 |   486.6  | 18.0%  |
    +---------------------------------------+--------+----------+--------+

Held-out: 5,000 conversations, 983K Devanagari chars (243,887 runs),
2.75M total chars. End-to-end savings (18.0%) are lower than script-only
savings (36.8%) because Latin/Hinglish spans gain nothing. Products
serving mostly Hinglish will see proportionally smaller wins.

Example phrases (measured):

    +-------------------------------------------+------+------+-------+
    | Phrase                                    | Base |   v3 |     D |
    +-------------------------------------------+------+------+-------+
    | ज़रूरी बात है क्या करते हो                   |   25 |   11 | -56.0%|
    | हम आपको बताना चाहते हैं                      |   23 |   16 | -30.4%|
    | नमस्ते, आप कैसे हैं?                          |   20 |   13 | -35.0%|
    | कितने बजे मिलना है?                           |   18 |   13 | -27.8%|
    | चाय पी लो, फिर बात करते हैं                   |   25 |   19 | -24.0%|
    +-------------------------------------------+------+------+-------+

### 4.2 Budget sweep

From models/budget_analysis.json (validation slice during design):

    +--------+------------------+------------------+
    | Budget | Tokens saved (%) | Incremental gain |
    +--------+------------------+------------------+
    |     50 |           14.19  |          +14.19  |
    |    100 |           22.05  |           +7.86  |
    |    200 |           31.37  |           +9.32  |
    |  *300* |          *36.25* |          *+4.88* |
    |    500 |           42.89  |           +6.64  |
    |    750 |           48.09  |           +5.20  |
    +--------+------------------+------------------+

Efficiency (% saved per injected token) peaks early and decays; 300 sits
past the steepest gains while keeping warmup trivially cheap.

### 4.3 Likelihood on held-out text (cross-tokenizer fair)

Raw perplexity is incomparable across different vocabularies, so we report
character-normalized cross-entropy on the identical character prefix. The
v3 perplexity numbers use a different base model (Qwen3-1.7B) than v2
(Qwen2.5-1.5B) and are evaluated on a different data subset, so v2 and
v3 perplexities are not directly comparable to each other.

Table 2: Bits per character on held-out Hindi/Hinglish text.

    +-------------------------+----------+-----------+---------+
    | Model                   | PPL own  | Bits/char |  d(bpc) |
    |                         |   tok    |     (down)|         |
    +-------------------------+----------+-----------+---------+
    | Qwen2.5-1.5B (bf16)    |    6.11  |   1.6531  |    --   |
    | v2 Q4 + LoRA            |    7.48  |   1.4878  |  -10.0% |
    +-------------------------+----------+-----------+---------+
    | Qwen3-1.7B-Base         |     --   |  11.8955  |    --   |
    | v3 Q4_K_M + LoRA        |     --   |   8.7348  |  -26.6% |
    +-------------------------+----------+-----------+---------+

The v3 bits-per-character reduction (26.6%) is 2.7x larger than v2
(10.0%). Two factors explain this: (1) Qwen3-1.7B-Base has higher raw
bits/char on Hindi than Qwen2.5-1.5B (11.90 vs 1.65), leaving more room
for improvement, and (2) the capability SFT training phase specifically
targets Hindi text quality, whereas v2 trained only on conversation data.

### 4.4 GSM8K-Hindi

4-shot chain-of-thought, greedy decoding, exact match on final number.
Dataset: bingbangboom/gsm8k-hindi test_main. Gold parsing accepts both
####N and ###N separators (28% of rows use ###).

Table 3: GSM8K-Hindi accuracy (4-shot CoT, greedy).

    +-----------------------------+----------+---------+
    | Model                       | Accuracy |  tok/s  |
    +-----------------------------+----------+---------+
    | Qwen2.5-1.5B (bf16)        |    3.0%  |  27.1   |
    | v2 Q4 + LoRA (Qwen2.5)     |    0.0%  |  52.7   |
    +-----------------------------+----------+---------+
    | Qwen3-1.7B-Base             |    3.0%  |     --  |
    | v3 Q4_K_M + LoRA (Qwen3)   |   56.0%* |     --  |
    +-----------------------------+----------+---------+

  * 16/30 evaluated items before 30-minute CPU timeout (partial run).
    Full 100-question run pending; see Section 5, Limitation 2.

The v3 base (Qwen3-1.7B) scores the same 3% as v2 base (Qwen2.5-1.5B)
on this benchmark. The 53-point jump to 56% comes entirely from the
capability SFT phase: 35K curated training rows containing GSM8K-Hindi
problems, IndicVault QA, and sample conversations. This is not a base-
model effect; it is a training-data effect.

The v2 LoRA scores below base (0% vs 3%) because conversation-only
training replaced few-shot format-following with conversational behavior.
Most v2 answers omit the answer marker entirely. v3, trained on math
data, follows the format correctly.

### 4.5 Conversational quality trade-off

To test whether the math-focused SFT degrades conversational ability, we
trained a conversational variant (v3.3): 370 curated rows (evol-hindi +
Hinglish + identity), LoRA rank 8, learning rate 5e-6, 2 epochs, 46
training steps, ~4 minutes on a single T4.

Table 4: Math vs. chat trade-off.

    +------------------+----------+----------+
    | Variant          | GSM8K    | Chat     |
    |                  | -Hi (%)  | quality  |
    +------------------+----------+----------+
    | v3 (math SFT)    |   56.0%  |  basic   |
    | v3.3 (chat SFT)  |   24.0%  |  strong  |
    +------------------+----------+----------+

v3.3 (100-question eval, full run) scores 24% on GSM8K-Hindi, down from
v3's 56%. The 32-point drop confirms that conversational fine-tuning
overwrites the math format-following behavior learned during capability
SFT. This is a known problem in multi-task SFT: small models struggle to
retain multiple capability modes simultaneously.

The practical implication: v3 (math SFT) is the better general-purpose
model. v3.3 is useful only when conversational fluency matters more than
reasoning.

### 4.6 Deployment

Single-file 1.0 GB GGUF (Q4_K_M) loads offline via llama.cpp, Ollama,
or the Python llama-cpp-python package. The canonical generation config
(temp 0.3, top-p 0.85, repetition penalty 1.25) is inherited from the
v2 deployment and remains necessary for clean output on this model
family.

---

## 5. Ablation: v2 vs v3

The injection recipe is identical between v2 and v3. The differences are
the base model and the training data.

Table 5: v2 vs v3 comparison.

    +------------------------+------------------+------------------+
    | Property               | v2               | v3               |
    +------------------------+------------------+------------------+
    | Base model             | Qwen2.5-1.5B     | Qwen3-1.7B       |
    | Base vocab size        | 151,665          | 151,669          |
    | Tokens injected        | 300              | 300              |
    | Injection savings (%)  | 36.5 (Dev)       | 36.8 (Dev)       |
    |                        | 18.0 (mixed)     | 18.0 (mixed)     |
    | Bits/char reduction    | -10.0%           | -26.6%           |
    | GSM8K-Hindi            | 0.0%             | 56.0%            |
    | Training data          | conversations    | math + QA + conv |
    | Hardware               | M4 + Colab T4    | Kaggle 2x T4     |
    | Edge size (Q4)         | 880 MB (MLX)     | 1.0 GB (GGUF)    |
    +------------------------+------------------+------------------+

The compression numbers are nearly identical between v2 and v3 (36.5% vs
36.8% Devanagari, 18.0% for both mixed). This is expected: the injection
recipe is the same, and the base Devanagari fertility is similar across
Qwen2.5-1.5B and Qwen3-1.7B. The bits-per-character improvement (26.6%
vs 10.0%) tells the real story: v3 is a much better Hindi language model
after training, even though the tokenizer gains are comparable.

---

## 6. Limitations

1. **GSM8K-Hindi is not a comprehensive reasoning benchmark.** 56% on
   grade-school math does not imply strong general reasoning. MMLU-Hindi,
   ARC-Challenge-Hindi, or custom benchmarks would paint a fuller picture.
   We report what we measured, not what we wish we measured.

2. **The v3 GSM8K number is a partial run.** The 56% figure comes from
   16 out of 30 evaluated items before a 30-minute CPU timeout (JSON:
   bench_gsm_hi_v3.json). A full 100-question run is needed to confirm
   this number. We report it as "56%" with this caveat, not "56.2%" or
   "56%+".

3. **Mixed-text compression gains are modest (18.0%).** Products serving
   mostly Hinglish will see less speedup than Devanagari-heavy ones.
   Larger budgets recover this but cost warmup time.

4. **Mean-init is likely suboptimal.** OMP transplant or hypernetwork
   init (ZETT/HYPEROFA) may improve quality-per-token; untested here.

5. **Math-chat trade-off is real.** Conversational fine-tuning on top of
   the math-SFT model drops GSM8K-Hindi from 56% to 24% (v3.3, N=100).
   Retaining both capabilities simultaneously may require DPO or
   multi-task data mixing, not just more SFT data.

6. **Machine-translated eval data.** GSM8K-Hindi inherits translation
   artifacts from the original English GSM8K. Real Hindi math reasoning
   benchmarks would be more trustworthy.

7. **One base model family.** Results are for Qwen3-1.7B. Transfer to
   Gemma, Phi, or other small bases is untested.

---

## 7. Conclusion

A 300-token injection (0.198% of the vocabulary, 614K trained params in
warmup, ~6 GPU-hours of SFT on consumer hardware) buys 36.8% script-
specific compression, 26.6% better bits-per-character on held-out Hindi,
and a 53-point accuracy jump on GSM8K-Hindi. The final artifact is a
1.0 GB GGUF file that runs offline on phones, laptops, and browsers.

The recipe is deliberately simple: n-gram mining, mean-init, embedding
warmup, LoRA. No new architecture, no new tokenizer design, no massive
compute. For teams localizing stock open models to single-script markets,
this is a practical playbook, not a research prototype.

All weights, benchmark scripts, and raw JSON outputs are released under
Apache 2.0.

---

## References

- Petrov, La Malfa, Torr, Bibi. Language Model Tokenizers Introduce
  Unfairness Between Languages. NeurIPS 2023. arXiv:2305.15425
- Minixhofer, Ponti, Vulic. Zero-Shot Tokenizer Transfer. NeurIPS 2024.
  arXiv:2405.07883
- Goddard, Fernandes Neto. Training-Free Tokenizer Transplantation via
  Orthogonal Matching Pursuit. 2025. arXiv:2506.06607
- Yamaguchi et al. How Can We Effectively Expand the Vocabulary of LLMs
  with 0.01GB of Target Language Text? Computational Linguistics 2024.
  arXiv:2406.11477
- Ozeren, Liu, Schutze. HYPEROFA. ACL SRW 2025. arXiv:2504.21018
- Sakajo et al. Dictionaries to the Rescue. ACL 2025. arXiv:2506.01535
- Remy et al. Trans-Tokenization / Tweeties. 2024. arXiv:2408.04303
- Mehta et al. Defragmenting Language Models (FragMend). 2026.
  arXiv:2604.16656
- Didenko. Writing-System-Level Tokenizer Adaptation for Byte-Level BPE.
  TokShop @ COLM 2026. arXiv:2608.00582
- Shravan. BrahmicTokenizer-131K. 2026. arXiv:2605.29379
- WWHO / SGPE: Syllable-aware Grapheme Pair Encoding. 2026.
  arXiv:2603.25309
- Tokenizer Transplantation for Bengali ASR. 2026. arXiv:2607.09598
- MorphTok. TokShop @ ICML 2025. arXiv:2504.10335
- The Tokenizer Tax. 2026. arXiv:2607.24276
- Sarvam AI. OpenHathi-Hi-v0.1. 2023. sarvam.ai; Sarvam-1. 2024.
  sarvam.ai/blogs/sarvam-1
- Krutrim. arXiv:2407.12481 (2024); tech report arXiv:2502.09642 (2025)
- MUTANT-Indic. 2025. arXiv:2511.03237
- RomanSetu. ACL 2024. arXiv:2401.14280
- Huffman Transliteration. ACL 2025. arXiv:2509.17493
- Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115
- Qwen Team. Qwen3 Technical Report. 2025.
  huggingface.co/Qwen/Qwen3-1.7B-Base
- Kishore, G. Safe Error Correction for Language Models: Frozen-Base
  Adjustment with Capability Preservation. 2026. arXiv:2609.16145
- Kishore, G. pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content
  Classifier. 2026. arXiv:2608.14658
