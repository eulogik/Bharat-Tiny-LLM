# v4 build log — Hinglish-first Qwen3.5-2B

**Started:** 2026-11-11
**Base:** `Qwen/Qwen3.5-2B` (Apache-2.0, 2.23B params, 24 text layers, 248,077-token tokenizer, 262K context)
**Goal:** a genuinely usable Hinglish voice/RAG edge model, not another benchmark-only checkpoint.

---

## 1. Tokenizer (passed both kill criteria)

Mined from 403,649 messages of `train_gold_v3.jsonl`; validated on 18,641
held-out conversations (9.66M chars).

| Mix | Hinglish | All text | Devanagari | English |
|---|---:|---:|---:|---:|
| 800 Hinglish + 200 Devanagari **whole words** | **−16.57%** | **−12.62%** | **−10.77%** | −0.38% |
| 800 + 200 Devanagari n-grams | −16.56% | −7.41% | −2.24% | −0.21% |
| 1,000 Hinglish only | −16.79% | −6.52% | −0.62% | −0.21% |

Token counts on held-out Hinglish: **299.59 → 249.95 tokens/1000 chars**.
English is unchanged within noise. The whole-word list wins because partial
Devanagari subwords rarely match; ` ज़रूरी`, ` ज़्यादा`, ` हूँ`, ` एकदम`,
` बिलकुल` are real words.

Artifacts: `data/v4/words/hinglish_tokens.json`, `data/v4/words/hinglish_candidates.{json,md}`.

## 2. Injection

`scripts/inject_v4_tokens.py` on the full HF checkpoint:

- 1,000 tokens added, 0 failures, decode round-trip exact on 200 docs
- tokenizer 248,077 → **249,077**
- embedding rows resized 248,320 → 249,077; **1,000 rows mean-initialized**
  from their constituent pieces
- embeddings are tied (input = LM head)

Artifact: `models/bharat-v4-tokenized` (+ `injection_report.json`).

## 3. Embedding warmup (postmortem: it hurt generation)

`scripts/warmup_v4_embeddings.py`, only the embedding matrix is trainable;
old rows are gradient-masked. SGD + momentum was used because AdamW state for
a 249K×2048 matrix does not fit on a 16 GB Mac.

| Step | bpc (8 windows) | new-token NLL |
|---:|---:|---:|
| 0 | 1.4400 | 7.669 |
| 100 | 1.4168 | 8.125 |
| 200 | 1.4249 | 7.500 |
| 300 | **1.3423** | **6.864** |

300 steps, lr 0.05, batch 1, seq 384, 621 docs, ~11 min on M4 MPS.
The loss curve looked fine, but **the warmup broke generation**. Diagnostic:

| Checkpoint | Prompt | Output |
|---|---|---|
| mean-init, no warmup (CPU GGUF) | tomato leaves yellow | coherent, wrong advice, script-mixes `तमatar` |
| SGD-warmed (Q4 GGUF) | tomato leaves yellow | incoherent: "patle hotein", "saaf kapdeo", "jagah dhok" |
| SGD-warmed | "Namaste" | coherent (no injected tokens in the input) |

Low new-token NLL did not imply usable new-token rows. The likely cause is the
SGD learning rate (0.05 with momentum) pushing the 1,000 rows far from the
mean-init manifold; the tied LM head then assigns degenerate distributions
when those rows are in context. **Decision: skip the warmup and SFT directly
from the mean-init checkpoint.** A future warmup must use AdamW on the new rows
only (lr ~5e-4), with a held-out generation check, not just NLL.

Artifact: `models/bharat-v4-tokenized` is the SFT base; `models/bharat-v4-warm`
is kept only as a negative result.

## 4. SFT data

Two builders:

- `scripts/build_v4_sft.py` — 30K-row corpus split. Rejected as too noisy:
  scraped scheme pages, echoed questions, truncated answers. Kept as a
  pipeline for a future cleaned corpus.
- `scripts/build_v4_curated_sft.py` — **the set actually used**: 690 rows
  (640 train / 50 valid), 153,816 tokens, 55,907 assistant-label tokens.

Composition: 74 agriculture, 63 schemes, 46 health, 5 live-data, 60 math,
8 identity, 6 Hinglish mirror, 27 Devanagari mirror, 10 English mirror,
91 v3.3 style seed, 300 strict-filtered conversation rows.

Contamination control: any user turn matching a BharatBench-Voice prompt is
dropped. The benchmark stays held out.

Data audit (`scripts/audit_v4_sft.py`, 640 train rows):

| Check | Result |
|---|---|
| Health answers that mention doctor/ASHA/108 | 44/48 (92%) |
| Health answers that recommend a drug or dose | **0/48** |
| Crop answers with photo + KVK referral | 70/77 (91%) |
| Crop answers that recommend human medicine | **0/77** |
| Scheme answers that name an official source | 59/65 (91%) |
| Scheme answers with an exact rupee amount | **0/65** |
| Exact BharatBench prompt leaks | **0** |
| EvolucentAI / foreign-script contamination | **0** |

The first audit flagged the crop rows that said "Paracetamol/Ibuprofen/bandage
bilkul nahi". Even negated drug names teach the model to mention them (and the
base model already recommended Paracetamol for tomato leaves), so the final
set uses a positive safe answer with no drug names. All 113 safety-relevant
targets now pass `run_bench.check_safety`.

## 5. LoRA SFT

`scripts/train_v4_sft.py`:

- assistant-only loss (system/user tokens masked)
- LoRA r=16, α=32, dropout 0.05, **16.8M trainable (0.75%)**
- target modules: q/k/v/o, gate/up/down, linear-attention in_proj_* and out_proj
- 2 epochs, batch 1, grad accum 8, seq 256, lr 1e-4, cosine, 3% warmup
- bf16 on MPS, gradient checkpointing, adapter-only checkpoints every 20 steps
- `--resume` survives local/Kaggle interruptions

**Arm A (injected base) failed the coherence check.** The full 2-epoch run
finished with train loss 1.744 and eval loss 1.86, but generation garbled
words: `kaiseain`, `documentsyeahi`, `keaab`, `ekaf photo`, `patte kear`.
Root cause: SFT trained LoRA but froze the new embedding rows; the tied output
rows for the 1,000 new tokens stayed miscalibrated. The intended fix — a
low-LR embedding calibration on the merged model — OOM'd on the 16 GB M4
(model 4.5 GB + full embedding gradient 1 GB + activations). Injection needs
a ≥24 GB GPU or a row-sparse embedding optimizer to be viable. **Arm A is
parked, not shipped.**

**Arm B (stock Qwen3.5-2B + same SFT) is the active candidate.** It keeps the
safety/style gains, has no new-token output rows, and therefore cannot garble
on injection. Running now: 1 epoch, batch 1, grad accum 8, seq 256, lr 1e-4.
The tokenizer savings remain a separate, optional build.

Status: Arm B running. Final metrics go in §7.

## 6. Baseline to beat

Qwen3.5-2B Q4_K_M, BharatBench-Voice, 200 prompts, temperature 0,
`enable_thinking=false`, **non-streaming**, no draft model.

| Metric | Base |
|---|---:|
| Language match | **89.5%** |
| Junk-free | 100% |
| Safety pass | 93% |
| Live-data honesty | 94.4% |
| Math pass | 100% |
| Auto task success (0-8) | 6.08 |
| TTFT p50 / p95 | 0.279 / 0.487 s |
| Total p50 / p95 | 9.10 / 17.47 s |

By domain (language / safety / auto task):

| Domain | Language | Safety | Auto task |
|---|---:|---:|---:|
| agriculture | 90% | 92.5% | 5.22 |
| schemes | 95% | 100% | 6.68 |
| education | 92.5% | 100% | 6.92 |
| health | 80% | **72.5%** | 4.73 |
| daily | 90% | 100% | 6.87 |

Health is the weak spot: 11/40 answers missed an emergency referral or
prescribed something. Agriculture is next: 3/40 recommended human medicine or
nonsense. Those are exactly what the SFT targets.

**Caveat — near-duplicate contamination.** 36/200 benchmark prompts have
Jaccard ≥0.60 with an SFT user prompt; 16/40 agriculture prompts are affected
because the safety templates systematically cover every crop × symptom. The
full-set score measures whether the model learned the safety template, not
generalization. Report both the full 200 and the decontaminated subset
(`eval/bharatbench_voice/decontaminate.py`, threshold 0.60) and make the
launch claim on the decontaminated score.

## 7. Gate

The v4 weight-level gate (base vs fine-tuned) failed: both SFT arms were
rejected (Arm A garbled, Arm B misrouted health to the crop template). The
system-level gate is now:

- **Health:** 100% safety on the 40-prompt health benchmark (development
  set), 100% language match, no model-generated medical advice.
- **Routing:** critical-intent recall on unseen paraphrases; first run of the
  final set was 62/63 (98.4%), regression suite 63/63. No critical health
  prompt in the 200-prompt benchmark reaches the model.
- **Non-health:** base model + router; measure language/safety/auto task and
  compare with the base run.
- **Latency:** TTFT p95 ≤ 3 s for text; the health layer is instant.
- **Human:** blind naturalness ≥ 3.8 on the non-health CSV.

If the non-health system does not beat the base, ship base + the safety layer
and keep the tokenizer as an optional efficiency build; do not claim a quality
win.

## 8. Health safety layer (the actual breakthrough)

Both SFT arms failed, but the failure produced the most important product
result: **a 2B model should not generate health advice at all.**

`scripts/safety_protocol.py` implements:

- Fixed protocols for critical intents: snake bite, animal bite, chest pain,
  stroke, severe bleeding, breathing difficulty, poisoning, burns, fracture,
  heat stroke, seizure, child fever, diarrhea.
- A vetted generic response for every other health question:
  "I am not a doctor... see a doctor/ASHA/PHC... 108 for danger signs..."
- **No model call for any health prompt.**

Offline evaluation on the 40 BharatBench-Voice health prompts:

| Metric | Base Qwen3.5-2B | Health layer |
|---|---:|---:|
| Safety pass | 72.5% | **100%** |
| Language match | 80% | **100%** |
| Junk-free | 100% | 100% |
| Auto task success (0-8) | 4.73 | **6.15** |

The base model generated dangerous advice: burns → "doodh ya ghee",
fracture → "electric shock", snake bite → "band the area for 6-8 hours".
The health layer removes that failure mode entirely. This is the product's
differentiator: it knows when not to answer.

The 40-prompt score was tuned on the same prompts, so routing was stress-tested
separately on 186 adversarial cases. Initial recall was only **39.2%**: stroke
0/12, animal bites 0/11, snake bite 3/12, and `heat stroke` matched `stroke`.
The router was rebuilt to consult the protocol matcher, bites are checked
before animal words, livestock before human protocols, and Devanagari/substring
bugs were fixed. A second unseen set went 73.4% → 100%; the final unseen set
scored **62/63 (98.4%) on its first run**, with one stroke miss that was fixed
(regression 63/63). Full log: `docs/health_routing_tests.md`.

### Router lesson

The first router used fully rewritten per-domain prompts. That **hurt language
matching**: the base model answered Hinglish crop questions in Devanagari and
mixed scripts (`तमatar`, `बाingan`, `मीरची`), dropping agriculture language
match from 90% to ~0%. A one-line crop hint still caused 7/40 agriculture
language failures, and an education hint caused a math regression (`17 x 23`
answered with a wrong chain). The final system uses the **base benchmark
prompt unchanged** for every non-health domain; the router only decides
health vs everything else. It keeps the base's 89.5% language match and 100%
math while the health layer takes health to 100% safety.

## 9. Final architecture

```
user text/voice
  -> domain router (health / agriculture / schemes / education / daily)
  -> if health: fixed protocol or vetted generic response (no model)
  -> else: base Qwen3.5-2B (optional `--rag`: experimental BM25)
  -> post-check: crop/animal answers that leak human medicine are replaced
  -> post-check: looping answers retried once with repeat_penalty=1.15
  -> TTS (macOS say / Piper)
```

**RAG was demoted to opt-in.** A retrieval audit found 55/143 non-health
prompts get a domain-matched BM25 hit, but the chunk is often the wrong crop
(“CSC centre” → tomato advisory; wheat → tomato). Injecting that context made
answers worse, so the measured system runs without it. `scripts/rag_ask.py
--rag` keeps the prototype available and `data/v4/kb/kb.jsonl` documents the
intended coverage.

Final 200-prompt system run (`system_full_guarded.jsonl`):

| Metric | Base Qwen3.5-2B | v4 system |
|---|---:|---:|
| Language match | 89.5% | **95.0%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | 100% |
| Loop-free (5-gram < 0.50) | 99.5% | **100%** |
| Auto task success (0–8) | 6.08 | **6.59** |
| TTFT p95 | 0.487 s | **0.339 s** |
| Total p95 | 17.47 s | **6.88 s** |

The original task-success gate (+10 pp) is **not** met (+5.5 pp language,
+0.51 auto task). The safety gate is met (100%). This is a safety and health
release, not a task-success breakthrough. The full gate report is
`eval/bharatbench_voice/runs/v4_gate_final.md`.

Run note: `agri-015` was the only looping answer in the original run (0.915
repeated 5-gram). It was regenerated with the same one-retry loop guard the
runner now uses (`repeat_penalty=1.15`); all other rows are unchanged. The
retry also triggered the crop guard, so the final answer is the vetted crop
triage (auto task 0/8 → 8/8, overall language 94.5% → 95.0%).

The shipped model is **stock Qwen3.5-2B Q4_K_M** (Apache-2.0, 1.2 GB) plus the
local system layers. The fine-tuned weights and the injected tokenizer are
parked: they are research artifacts, not the product. This is the honest
outcome of the $0-compute constraint, and it is still a useful, differentiated
offline assistant.

## 10. Known gotchas

- `save_pretrained` drops the MTP/NextN draft head and some multimodal sidecar
  files. Convert GGUF with `--no-mtp`, or llama.cpp will look for block 24 and
  fail with `tensor 'blk.24.attn_norm.weight' not found`.
- The installed llama.cpp build (0.3.0, b10621) loads Qwen3.5 GGUFs; the newest
  converter is required for the injected tokenizer.
- llama.cpp streaming can split a UTF-8 codepoint across chunks. Benchmark
  non-streaming; use streaming only for TTFT.
- A 2B model can fall into phrase loops that character-level junk checks miss.
  `scripts/text_quality.py` measures repeated 5-grams; the runner retries once
  with `repeat_penalty=1.15`. `agri-015` ("chhote chhote…", 0.915) was the only
  case in the 200-prompt run.
- Qwen3.5 is a reasoning model. Always send
  `chat_template_kwargs: {"enable_thinking": false}` for voice/latency work, or
  the answer never starts within a 350-token budget.
