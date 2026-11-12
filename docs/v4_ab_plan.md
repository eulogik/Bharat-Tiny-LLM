# v4 SFT A/B plan

**Why:** the SGD embedding warmup improved held-out NLL but broke generation
(see `docs/v4_progress.md` §3). The mean-init checkpoint is coherent, so the
first arm trains from it. If it still script-mixes or gives unsafe answers, the
second arm drops injection entirely and trains the same data on stock
Qwen3.5-2B.

## Arms

| Arm | Base | Tokenizer | SFT data | Purpose |
|---|---|---|---|---|
| **A** | `models/bharat-v4-tokenized` (mean-init) | 1,000 injected | 640 curated | quality + tokenizer savings |
| **B** | `Qwen/Qwen3.5-2B` (stock) | stock 248K | 640 curated | quality-only fallback |
| **C** (existing) | Qwen3.5-2B Q4 | stock | none | baseline to beat |

Arm B is the honest fallback: the tokenizer is an enabler, not the product. If
A is not clearly better than B on generation quality, ship B and keep the
tokenizer as an optional efficiency build.

## Commands

Arm A (running):

```bash
.venv-v4/bin/python scripts/train_v4_sft.py \
  --model models/bharat-v4-tokenized \
  --train data/v4/sft_curated/train.jsonl \
  --valid data/v4/sft_curated/valid.jsonl \
  --out models/bharat-v4-lora-meaninit \
  --epochs 2 --batch 2 --grad-accum 4 --seq 384 \
  --lora-r 16 --lora-alpha 32 --lr 1e-4 --dtype bf16 \
  --eval-steps 20 --save-steps 20 --resume
```

Arm B (only if A fails):

```bash
.venv-v4/bin/python scripts/train_v4_sft.py \
  --model "/Volumes/KIOXIA 1TB/hf_q35_full" \
  --train data/v4/sft_curated/train.jsonl \
  --valid data/v4/sft_curated/valid.jsonl \
  --out models/bharat-v4-lora-base \
  --epochs 2 --batch 2 --grad-accum 4 --seq 384 \
  --lora-r 16 --lora-alpha 32 --lr 1e-4 --dtype bf16 \
  --eval-steps 20 --save-steps 20 --resume
```

## Evaluation

1. Merge each adapter into its base (`scripts/merge_v4_lora.py`).
2. Convert to GGUF with `--no-mtp` (the MTP head is dropped by
   `save_pretrained`), quantize Q4_K_M.
3. Run `eval/bharatbench_voice/run_bench.py` **non-streaming**, 200 prompts,
   `enable_thinking=false`.
4. Run `compare.py --base <C> --candidate <A or B>`.
5. Blind-score the CSV for task success and naturalness on at least 60 prompts.
6. Sample 20 open-ended prompts manually; script mixing, loops, and unsafe
   advice are automatic rejects regardless of the aggregate score.

## Decision rule

- **Ship A** only if it beats C by ≥10 pp language match, safety ≥ C, no loops/
  script mixing in the manual sample, and it is not worse than B on the same
  metrics. Then the tokenizer savings (16.57% Hinglish) can be stated.
- **Ship B** if A fails but B is coherent and safer than C. Do not claim
  tokenizer savings for the shipped checkpoint; keep them as an optional build.
- **Ship C + RAG** if both A and B fail. The product becomes base + retrieval +
  prompt + TTS, which is still useful and honest.

## Result (2026-11-12)

**Arm A failed the coherence check.** After the full 2-epoch SFT, the injected
model produced garbled words: `kaiseain`, `documentsyeahi`, `keaab`,
`ekaf photo`, `patte kear`. Root cause: SFT trained LoRA but froze the new
embedding rows; the tied output rows for the 1,000 new tokens remained
miscalibrated. A low-LR embedding calibration (the intended fix) OOM'd on the
16 GB M4 (model 4.5 GB + full embedding gradient 1 GB + activations). The
injection path needs a ≥24 GB GPU or a row-sparse embedding optimizer to be
viable. **Arm A is parked, not shipped.**

Arm B (stock Qwen3.5-2B + the same 640-row SFT) is now the active candidate.
It keeps the safety/style gains, has no new-token output rows, and therefore
cannot garble on injection. The tokenizer savings remain a separate,
optional build for hardware that can support the calibration.

## Result — Arm B also rejected (2026-11-12)

Arm B trained cleanly (train loss 1.741, 1 epoch) and produced coherent text,
but it **destroyed domain routing**: with the health prompt it answered a
child-diarrhoea question with the crop template ("paudhon ke upar aur neeche
... photo bhejiye"), and it told a snake-bite victim to band the area. The
640-row set is too repetitive and too small for a 2B model; the crop template
is the most frequent pattern and it overgeneralizes.

**Neither SFT arm is shipped.** The resolution is system-level:

- `scripts/domain_prompt.py` routes health vs agriculture before generation.
- `scripts/safety_protocol.py` returns fixed, vetted health responses and
  **never calls the model for a health question**.
- Health result on the 40-prompt set: **100% safety, 100% language match**
  (base: 72.5% / 80%), auto task 6.15 vs 4.73.
- Non-health questions use the stock base model + RAG + the domain prompt.

This is the honest outcome: the model is not the breakthrough; knowing when
not to answer is. The tokenizer and SFT work remain documented research
artifacts, not shipped weights.



## Cost

Each arm is ~35 min on the M4 (640 rows, 2 epochs, batch 2, seq 384). GGUF
export/quantize is ~2 min. A full 200-prompt non-streaming benchmark is ~17 min
per model. Total for both arms: ~2 hours, $0.
