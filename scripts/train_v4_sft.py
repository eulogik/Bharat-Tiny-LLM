#!/usr/bin/env python3
"""LoRA SFT for Bharat-Tiny-LLM v4 on Qwen3.5-2B.

Designed for one Kaggle T4 (16 GB) or a local Apple Silicon machine. The base
model is only 2B, so fp16 LoRA fits without 4-bit quantization; that avoids
bitsandbytes compatibility risk with the new hybrid Qwen3.5 architecture.

Prerequisites
-------------
  pip install -U "transformers>=4.57.6" peft trl datasets accelerate

Usage (Kaggle)
--------------
  python scripts/train_v4_sft.py \
    --model Qwen/Qwen3.5-2B \
    --train data/v4/sft/train.jsonl \
    --valid data/v4/sft/valid.jsonl \
    --out /kaggle/working/bharat-v4-lora \
    --epochs 2 --batch 1 --grad-accum 16 --seq 2048

  # smoke test
  python scripts/train_v4_sft.py --max-steps 20 --max-rows 200

After training, merge the adapter and export GGUF:
  python scripts/merge_v4_lora.py --base Qwen/Qwen3.5-2B \
    --adapter /kaggle/working/bharat-v4-lora --out /kaggle/working/bharat-v4-merged
  python convert_hf_to_gguf.py ... --outtype f16
  llama-quantize ... q4_k_m
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def load_base(model_id: str, dtype: str):
    import transformers

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    last_exc = None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            print(f"Loading {model_id} with {loader_name} ({dtype}) ...", file=sys.stderr)
            return loader.from_pretrained(
                model_id, dtype=torch_dtype, trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  {loader_name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise RuntimeError(f"could not load {model_id}") from last_exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--train", default="data/v4/sft_curated/train.jsonl")
    ap.add_argument("--valid", default="data/v4/sft_curated/valid.jsonl")
    ap.add_argument("--out", default="models/bharat-v4-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out, if any")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenize and report stats without loading the model")
    ap.add_argument("--full-loss", action="store_true",
                    help="train on the whole conversation instead of assistant tokens only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    dtype = args.dtype
    if dtype == "auto":
        dtype = "bf16" if (torch.cuda.is_bf16_supported() or torch.backends.mps.is_available()) else "fp16"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files={"train": args.train, "valid": args.valid})
    if args.max_rows:
        ds["train"] = ds["train"].select(range(min(args.max_rows, len(ds["train"]))))
        ds["valid"] = ds["valid"].select(range(min(max(20, args.max_rows // 10), len(ds["valid"]))))

    def preprocess(row):
        msgs = row["messages"]
        if args.full_loss:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            enc = tok(text, truncation=True, max_length=args.seq, add_special_tokens=False)
            enc["labels"] = enc["input_ids"].copy()
            return enc

        # Assistant-only loss: mask the system/user prompt. The curated set is
        # single-turn, so the last assistant message is the target.
        last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompt = tok.apply_chat_template(
            msgs[:last], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = msgs[last]["content"].strip() + (tok.eos_token or "")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + a_ids)[: args.seq]
        labels = ([-100] * len(p_ids) + a_ids)[: args.seq]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    remove = list(ds["train"].column_names)
    ds = ds.map(preprocess, remove_columns=remove, desc="tokenizing")
    ds = ds.filter(
        lambda r: len(r["input_ids"]) >= 16
        and sum(1 for x in r["labels"] if x != -100) >= 4
    )
    print(f"Train rows: {len(ds['train'])}, valid rows: {len(ds['valid'])}", file=sys.stderr)
    print(f"Train tokens: {sum(len(x) for x in ds['train']['input_ids']):,}", file=sys.stderr)
    lengths = [len(x) for x in ds["train"]["input_ids"]]
    label_tokens = sum(1 for row in ds["train"]["labels"] for x in row if x != -100)
    stats = {
        "train_rows": len(ds["train"]),
        "valid_rows": len(ds["valid"]),
        "train_tokens": sum(lengths),
        "label_tokens": label_tokens,
        "mean_tokens": round(sum(lengths) / len(lengths), 1),
        "max_tokens": max(lengths),
        "p95_tokens": sorted(lengths)[int(len(lengths) * 0.95) - 1],
    }
    if args.dry_run:
        print(json.dumps(stats, indent=2))
        return

    model = load_base(args.model, dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    model.config.use_cache = False

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "out_proj",
    ]
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)", file=sys.stderr)

    collator = DataCollatorForSeq2Seq(
        tok, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
    )
    out = Path(args.out)
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    # Transformers 5 removed warmup_ratio; compute the step count ourselves.
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = max(1, len(ds["train"]) // (args.batch * args.grad_accum))
        total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(total_steps * 0.03))
    # On MPS, bf16 is natively supported and much faster than fp32.
    use_bf16 = (dtype == "bf16")
    use_fp16 = (dtype == "fp16")
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        save_only_model=True,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        data_collator=collator,
    )
    resume = None
    if args.resume:
        ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume = str(ckpts[-1])
            print(f"Resuming from {resume}", file=sys.stderr)
        else:
            print("No checkpoint found; starting from scratch", file=sys.stderr)
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out))
    tok.save_pretrained(out)

    report = {
        "model": args.model,
        "train_rows": len(ds["train"]),
        "valid_rows": len(ds["valid"]),
        "train_tokens": sum(len(x) for x in ds["train"]["input_ids"]),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 3),
        "train_metrics": train_result.metrics,
        "args": vars(args),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
