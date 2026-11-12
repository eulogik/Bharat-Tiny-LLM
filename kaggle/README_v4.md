# Bharat-Tiny-LLM v4 — Kaggle runbook

> **Status (2026-11-12): this runbook is for the parked fine-tuning research,
> not the shipped v4 system.** Both SFT arms were rejected (the injected model
> garbled; the stock-base model misrouted health to the crop template). The
> shipped v4 is stock Qwen3.5-2B + the local router, health safety layer and
> RAG. See `docs/v4_ab_plan.md` and `docs/v4_progress.md`. Keep this runbook if
> you want to retry SFT with a larger, more diverse dataset on a 24 GB GPU.

This is the $0-compute path: one Kaggle T4 (16 GB) is enough for the 2B model
with fp16 LoRA. No 4-bit quantization is required, which avoids
`bitsandbytes` compatibility risk with the new Qwen3.5 hybrid architecture.

## What to upload

1. This repository (or at least `scripts/`, `eval/`, `data/v4/`).
2. `data/v4/hinglish_tokens.json` — the 1,000-token injection list.
3. `data/v4/sft/train.jsonl` and `valid.jsonl` — 30K curated rows (~50 MB).
4. Optional: `train_gold_v3.jsonl` as a Kaggle dataset if you want to rebuild
   the SFT data on Kaggle. The checked-in 30K split is enough for the first run.

## Setup

```bash
!pip install -q -U "transformers>=4.57.6" peft trl datasets accelerate
!git clone https://github.com/eulogik/Bharat-Tiny-LLM.git
%cd Bharat-Tiny-LLM
```

Qwen3.5-2B is Apache-2.0; no HF token is needed.

## Step 1 — inject the Hinglish tokens

```bash
!python scripts/inject_v4_tokens.py \
  --model Qwen/Qwen3.5-2B \
  --tokens data/v4/hinglish_tokens.json \
  --whole-words-only \
  --out /kaggle/working/bharat-v4-tokenized \
  --dtype bf16
```

Expected: `old_tokenizer_len=248077`, `new_tokenizer_len=248914`,
`rows_mean_initialized=837`, round-trip OK. The script saves the tokenizer,
the resized model, and `injection_report.json`.

## Step 2 — warm up the new embedding rows

```bash
!python scripts/warmup_v4_embeddings.py \
  --model /kaggle/working/bharat-v4-tokenized \
  --corpus data/v4/sft_curated/train.jsonl \
  --eval-corpus data/v4/sft_curated/valid.jsonl \
  --out /kaggle/working/bharat-v4-warm \
  --steps 500 --batch 2 --seq 512 --lr 1e-3 \
  --dtype fp16 --device cuda
```

If the T4 runs out of memory, use `--batch 1 --seq 384`. The warmup only
optimizes the embedding matrix; old rows are gradient-masked. On a T4 (no
bf16), fp16 is the fast path; if you see NaNs, switch `--dtype fp32` and keep
`--batch 1` (the 2B model plus embedding gradients fits in 16 GB).

Target: new-token NLL falls by more than 50%, and bits/char recovers to within
5% of the base model. If it does not, stop and ship the base + RAG.

## Step 3 — LoRA SFT

```bash
!python scripts/train_v4_sft.py \
  --model /kaggle/working/bharat-v4-warm \
  --train data/v4/sft_curated/train.jsonl \
  --valid data/v4/sft_curated/valid.jsonl \
  --out /kaggle/working/bharat-v4-lora \
  --epochs 2 --batch 1 --grad-accum 8 --seq 1024 \
  --lora-r 16 --lora-alpha 32 --lr 1e-4 \
  --dtype fp16 --eval-steps 20 --save-steps 20 --resume
```

`--resume` picks up the latest `checkpoint-*` after a Kaggle or local
interruption; checkpoints are adapter-only (`save_only_model=True`), so they
cost ~70 MB each.

Smoke test first:

```bash
!python scripts/train_v4_sft.py --max-rows 50 --max-steps 10 \
  --out /kaggle/working/smoke --dtype fp16
```

Expected: ~640 train rows, ~56K assistant-label tokens/epoch, 10–20 minutes
for 3 epochs on one T4. The script prints trainable parameter count (target:
1–3% of 2B) and saves `training_report.json`. On a T4, fp16 is the fast path;
if you see NaNs, switch `--dtype fp32` (2B fp32 + LoRA still fits in 16 GB at
batch 1, seq 512).

## Step 4 — merge and export GGUF

```bash
!python scripts/merge_v4_lora.py \
  --base /kaggle/working/bharat-v4-warm \
  --adapter /kaggle/working/bharat-v4-lora \
  --out /kaggle/working/bharat-v4-merged \
  --dtype bf16

!git clone https://github.com/ggml-org/llama.cpp
# --no-mtp is required: save_pretrained drops the MTP draft head, so the
# converter must not reserve a 25th block for it.
!python llama.cpp/convert_hf_to_gguf.py /kaggle/working/bharat-v4-merged \
  --outtype f16 --no-mtp --outfile /kaggle/working/bharat-v4-f16.gguf
!llama-quantize /kaggle/working/bharat-v4-f16.gguf \
  /kaggle/working/bharat-v4-q4_k_m.gguf Q4_K_M
```

llama.cpp must be recent enough to support the Qwen3.5 hybrid architecture.
The existing local GGUF runs on the installed build, so use the same commit for
conversion.

## Step 5 — evaluate before shipping

```bash
# bits/char vs base, same character windows
!python scripts/bench_bpc.py \
  --corpus data/v4/sft/valid.jsonl \
  --windows 200 --window-chars 1024 --seq 2048 --dtype fp16 \
  --model Qwen/Qwen3.5-2B \
  --model /kaggle/working/bharat-v4-merged \
  --out /kaggle/working/bench_bpc_v4.json

# task benchmark against a running llama-server
!python eval/bharatbench_voice/run_bench.py \
  --base-url http://127.0.0.1:8096/v1 \
  --model bharat-v4-q4_k_m.gguf \
  --out /kaggle/working/bharatbench_v4.jsonl
```

Launch gate: v4 must beat Qwen3.5-2B base by **≥10 points** on Hinglish task
success, with safety ≥ base and TTFT ≤ 3 s. If it does not, ship the base with
prompting + RAG and keep the tokenizer as an optional efficiency build.

## Cost and time

| Step | Time on one T4 | Disk |
|---|---:|---:|
| Inject | 5–10 min | ~5 GB |
| Warmup (500 steps) | 20–40 min | ~5 GB |
| SFT (2 epochs) | 2–3 h | ~6 GB (adapter + checkpoints) |
| Merge + GGUF | 10–15 min | ~5 GB |
| Benchmarks | 1–2 h | small |

Total: one Kaggle session, well inside the 12-hour limit, $0.
