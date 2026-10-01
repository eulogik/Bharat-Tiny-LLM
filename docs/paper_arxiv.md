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

Injection alone reduces token count on pure Devanagari text by 36.9% and
on mixed Hinglish/Devanagari conversations by 17.7%. After training, the
model predicts held-out Hindi text at 26.6% fewer bits per character than
its base under identical character coverage. On GSM8K-Hindi (4-shot
chain-of-thought, greedy decoding, n=100), scores are dominated by
inference-stack effects: 2% under torch/fp16 versus 29% under
llama.cpp/Q4 for identical prompts and weights (chat-tuned variant:
35%). An early 9/16 partial run suggesting 56% did not reproduce under
either pinned protocol and is retracted (see Section 6). The whole
pipeline runs on consumer hardware (Kaggle 2x T4, zero cloud cost)
and ships as a 1.0 GB 4-bit GGUF file, downloadable from Hugging Face.

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
   (+0.198% of vocab) deliver 36.9% script-specific compression.
2. A two-phase recovery recipe (embedding-row warmup, then LoRA SFT on
   Hindi math and QA data) that recovers and exceeds base-model Hindi
   likelihood, measured via bits-per-character across different tokenizers.
3. A documented negative result on math SFT: 35K curated training
   rows move GSM8K-Hindi only within a stack-dominated 2-35% band
   (n=100, pinned protocols), with no version separation exceeding
   stack variation; an early 56% partial did not reproduce.
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
minimal budget (300 tokens, 0.198% of vocab) delivering 36.9% script-
specific compression, 26.6% better bits-per-character, a pinned-stack
GSM8K-Hindi study that retracts an early 56% partial, plus an honest
cross-tokenizer evaluation protocol and a fully open artifact.

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
Kaggle T4). The poison gate monitors validation loss on
chunks containing injected tokens and stops if loss diverges.

### 3.6 Phase 2: Capability SFT

LoRA fine-tuning on a curated Hindi reasoning corpus: 35,000 rows
containing GSM8K-Hindi (grade-school math), IndicVault (factual QA),
and sample conversations. LoRA rank 32, alpha 64, applied to all
attention and MLP layers. Distributed across 2x Kaggle T4 GPUs via
torchrun (DDP), batch size 4 per GPU (16 effective with gradient
accumulation), max sequence length 1024, cosine schedule.

| | v3 SFT | v3.3 SFT |
|---|---|---|
| Base | Qwen3 + 300 inj. | merged v3 |
| LoRA | r32 / a64 | r8 / a16 |
| Dropout | 0.05 (both) | 0.05 (both) |
| Targets | q,k,v,o + gate,up,down (both) | q,k,v,o + gate,up,down (both) |
| Data | 35,209 rows | 370 rows |
| Epochs (steps) | 2 (~4.4K) | 2 (46) |
| LR | 1e-4 | 5e-6 |
| Batch | 4/GPU, accum 2 (eff 16) | 4/GPU, accum 4 (eff 16) |
| Seq len | 1024 | 512 |
| Optim / sched | adamw_fused / cosine | adamw / cosine |
| Hardware | 2xT4 torchrun DDP | single T4 |

### 3.7 Merge and export

LoRA weights merged into base; full model exported to F16 GGUF, then
quantized to Q4_K_M. Final artifact: 1.0 GB single-file GGUF.

---

## 4. Results

All measurements use text held out from token selection. Fertility
(Table 1) is measured on 18,641 validation conversations
(data/processed/valid_gold.jsonl; JSON: docs/bench_fertility_v3.json);
v2 reference figures come from docs/bench_fertility.json (lines
300K-305K, 5,000 conversations). Scripts: scripts/bench_fertility.py,
scripts/bench_ppl.py, scripts/bench_gsm_hi.py. Raw JSON outputs:
docs/bench_fertility.json, docs/bench_fertility_v3.json,
docs/bench_ppl_v3.json, docs/bench_gsm_hi_v3.json,
docs/bench_gsm_hi_v3_torch.json, docs/bench_gsm_hi_v3_llama.json,
docs/bench_gsm_hi_v33_llama.json.

### 4.1 Compression (injection only, no training involved)

Table 1: Token compression under Qwen3-1.7B-Base tokenizer.

    +---------------------------------------+--------+----------+--------+
    | Metric                                |  Base  | +300 tok | Savings|
    +---------------------------------------+--------+----------+--------+
    | Tokens/1000 chars (Devanagari runs)   | 1042.9 |   658.3  | 36.9%  |
    | Chars/token (Devanagari runs)         |  0.959 |   1.519  |+58.4%  |
    | Tokens/1000 chars (full mixed convos) |  586.5 |   482.6  | 17.7%  |
    +---------------------------------------+--------+----------+--------+

Held-out: 18,641 conversations, 3.36M Devanagari chars (840,493 runs),
9.68M total chars. End-to-end savings (17.7%) are lower than script-only
savings (36.9%) because Latin/Hinglish spans gain nothing. Products
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
Dataset: bingbangboom/gsm8k-hindi test_main, first 100 rows. Gold
parsing accepts both ####N and ###N separators (24% of rows use ###).
Prompts, exemplars (train rows 0-3), and answer extraction are
byte-identical across runs; only the inference stack varies. JSONs:
bench_gsm_hi_v3_torch.json, bench_gsm_hi_v3_llama.json,
bench_gsm_hi_v33_llama.json.

Table 3: GSM8K-Hindi accuracy (4-shot CoT, greedy, n=100). Stacks are
not comparable across rows.

    +-----------------------------+----------+---------+
    | Model / stack               |    n     | Accuracy|
    +-----------------------------+----------+---------+
    | v3 / torch fp16 (2xT4)     |   100    |   2.0%  |
    | v3 / torch fp16 1-GPU (T4) |   100    |   2.0%  |
    | v3 / llama.cpp Q4 (CPU)    |   100    |  29.0%  |
    | v3.3 / llama.cpp Q4 (CPU)  |   100    |  35.0%  |
    +-----------------------------+----------+---------+

Two independent torch runs (2-GPU auto map and single-GPU) agree at
2.0%. Prior base-model (3%) and v2 (0%) figures used other stacks and
are omitted as incomparable. The 9/16 ad-hoc run once reported as 56%
used an unrecorded protocol and is retracted.

The headline finding is the stack gap, not the scores. Identical
prompts fed to identical weights produce degenerate question-echoing
loops under torch/fp16 but genuine reasoning attempts under
llama.cpp/Q4. The mechanism is unpinned (repetition-penalty defaults
were checked and ruled out), but the measurement stands across two
independent torch runs and full n=100 llama runs. Within a pinned
stack, versions barely separate: llama gives v3.3 (35%) over v3 (29%),
torch gives v3.4 (~13%, partial) over v3 (2%). Neither gap exceeds
noise plus stack variation. There is no evidence here that 35K rows of
math SFT bought robust GSM8K-Hindi ability, and no evidence of a
math-versus-chat trade-off in GSM scores either.

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
    | v3 (math SFT)    | 2 torch /|  basic   |
    |                  | 29 llama |          |
    | v3.3 (chat SFT)  | 35 llama |  strong  |
    | v3.4 (mixed SFT) | ~13*     |  weak    |
    |                  | torch    |          |
    | v3.5 (math x2,   |    ---+  | regress. |
    |   r=32)          |          |          |
    +------------------+----------+----------+

* v3.4 GSM is partial (MPS timeouts); shown for completeness, not
ranked. + v3.5 GSM not benched (MPS too slow for n=100); math failure
established by direct probes. Full curves and bench outputs:
v35_final_decision.json.

All GSM cells are n=100 except v3.4 (partial n=20/40). Within a pinned
stack, later versions score higher, not lower: llama gives v3.3 (35%)
over v3 (29%), torch gives v3.4 (~13%) over v3 (2%). Neither gap
exceeds noise plus stack variation, so there is no measurable GSM
trade-off from chat or mixed SFT here. The earlier frontier reading
(56% to 24% to 13%) mixed unpinned stacks with an irreproducible
partial and is withdrawn. What did replicate across stacks is the chat
degradation: v3.4 assistants echo the User: role on raw prompts, and
v3.5 regresses outright. The practical implication is unchanged: v3
remains the model to ship, now because later versions buy no measured
math gain while degrading chat, not because of a GSM trade-off.

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
    | Injection savings (%)  | 36.5 (Dev)       | 36.9 (Dev)       |
    |                        | 18.0 (mixed)     | 17.7 (mixed)     |
    | Bits/char reduction    | -10.0%           | -26.6%           |
    | GSM8K-Hindi            | 0.0% (old stack)| 29% llama, 2%    |
    |                        |                  | torch (n=100)    |
    | Training data          | conversations    | math + QA + conv |
    | Hardware               | M4 + Colab T4    | Kaggle 2x T4     |
    | Edge size (Q4)         | 880 MB (MLX)     | 1.0 GB (GGUF)    |
    +------------------------+------------------+------------------+

The compression numbers are nearly identical between v2 and v3 (36.5% vs
36.9% Devanagari, 18.0% vs 17.7% mixed). This is expected: the injection
recipe is the same, and the base Devanagari fertility is similar across
Qwen2.5-1.5B and Qwen3-1.7B. The bits-per-character improvement (26.6%
vs 10.0%) tells the real story: v3 is a much better Hindi language model
after training, even though the tokenizer gains are comparable.

---

## 6. Limitations

1. **GSM8K-Hindi is not a comprehensive reasoning benchmark.** A high
   score on grade-school math would not imply strong general reasoning.
   MMLU-Hindi, ARC-Challenge-Hindi, or custom benchmarks would paint a
   fuller picture. We report what we measured, not what we wish we
   measured.

2. **The v3 GSM8K headline (56%) is retracted.** The figure came from
   9 correct out of 16 evaluated items in an ad-hoc run whose protocol
   was never recorded (JSON: bench_gsm_hi_v3.json). Pinned
   re-measurement gives 2% (torch/fp16, n=100, two runs) and 29%
   (llama.cpp/Q4, n=100) on identical prompts and weights. Treat every
   GSM number in this paper as stack-pinned or incomparable.

   **Greedy exact-match GSM on 1.7B is stack-dominated.** Identical
   prompts and weights score 2% under torch but 29% under llama.cpp.
   The mechanism is unpinned (repetition-penalty defaults were checked
   and ruled out), but the measurement stands across two independent
   torch runs and full n=100 llama runs. Until the mechanism is
   understood, single-point GSM claims on small models, ours included,
   should be read as stack-plus-model measurements, and cross-stack
   rankings are invalid.

3. **Mixed-text compression gains are modest (17.7%).** Products serving
   mostly Hinglish will see less speedup than Devanagari-heavy ones.
   Larger budgets recover this but cost warmup time.

4. **Mean-init is likely suboptimal.** OMP transplant or hypernetwork
   init (ZETT/HYPEROFA) may improve quality-per-token; untested here.

5. **Math-chat trade-off is real, and mixed SFT does not fix it.**
   Conversational fine-tuning on top of the math-SFT model changes
   GSM8K-Hindi within noise under pinned stacks (llama: v3.3 35% vs
   v3 29%; torch: v3.4 ~13% vs v3 2%), while mixed math+chat SFT
   degrades chat outright: v3.4 (~17% math) has weak chat, and v3.5
   (~23% math after x2 upsampling, LoRA r=32) regresses outright,
   wrong arithmetic, incoherent Hindi identity answers, and
   resurfacing Thai/Arabic script junk under sampling controls that
   passed clean for v3.4. Stacking LoRA on LoRA-merged weights
   compounds format drift; more rank is not a substitute for clean
   data. Retaining both capabilities simultaneously likely requires
   DPO, a larger base, or continued pretraining, not more SFT mixing.

6. **Machine-translated eval data.** GSM8K-Hindi inherits translation
   artifacts from the original English GSM8K. Real Hindi math reasoning
   benchmarks would be more trustworthy.

7. **One base model family.** Results are for Qwen3-1.7B. Transfer to
   Gemma, Phi, or other small bases is untested.

---

## 7. Conclusion

A 300-token injection (0.198% of the vocabulary, 614K trained params in
warmup, SFT on Kaggle 2xT4) buys 36.9% script-
specific compression and 26.6% better bits-per-character on held-out
Hindi. The GSM8K-Hindi campaign ended in a retraction (an early 56%
partial did not reproduce) and a finding that stands on its own:
greedy exact-match scores on a 1.7B model are dominated by the
inference stack. The final artifact is
a 1.0 GB GGUF file that runs offline on phones, laptops, and browsers.

The recipe is deliberately simple: n-gram mining, mean-init, embedding
warmup, LoRA. No new architecture, no new tokenizer design, no massive
compute. For teams localizing stock open models to single-script markets,
this is a practical playbook, not a research prototype.

All weights, benchmark scripts, and raw JSON outputs are released under
Apache 2.0.

---

## Appendix

### A. Follow-up training curves

v3.4 (mixed math+chat, 43,887 rows at ~17% math, LoRA r=16, 1 epoch,
lr 2e-5, seq 1024, 2xT4 DataParallel) trained 611 min to final loss
1.2337 (JSON: v34_bench_report.json). v3.5 (same data with math
upsampled x2, LoRA r=32) was killed by the session limit at step
~3070/3082; train loss sat at ~0.9-1.0 with eval loss falling
1.327 to 1.217 across six evaluations (JSON: v35_final_decision.json).

### B. Qualitative samples

Live endpoint outputs (Featherless AI, temp 0.3, 2026-10-01), shown as
displayed after the demo scrubber: "भारत की राजधानी क्या है?" got
"भारत की राजधानी नई दिल्ली ।"; "17 * 23 kitna hota hai?" got
"17 * 23 = 391" plus "उत्तर: 391".

### C. Reproduction checklist

Fertility: scripts/bench_fertility.py. Likelihood: scripts/bench_ppl.py.
GSM8K-Hindi: scripts/bench_gsm_hi.py (MLX) and
scripts/bench_gsm_hi_v34.py (torch); pinned JSONs
bench_gsm_hi_v3_torch/llama and bench_gsm_hi_v33_llama. Training data:
eulogik/bharat-v3-mixed-sft-v34 (43,887 rows) and
eulogik/bharat-v3-chat-sft-v3 (370 rows) on the Hub.

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
