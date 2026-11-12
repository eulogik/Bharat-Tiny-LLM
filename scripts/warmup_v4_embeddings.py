#!/usr/bin/env python3
"""Warm up the new Hinglish embedding rows of an injected Qwen3.5-2B.

POSTMORTEM (2026-11-12): the original recipe here (SGD, lr 0.05, 300 steps)
improved held-out new-token NLL but produced incoherent generation
("patle hotein", "saaf kapdeo" on a tomato-leaf prompt). The mean-init
checkpoint without warmup is coherent. If you run this script, use a much
smaller LR (<= 5e-4) and ALWAYS run a generation check, not just NLL. The v4
SFT now starts from the mean-init checkpoint.

The injection script mean-initializes new rows, but the model has never seen
them as atomic tokens. This script freezes every parameter except the input
embedding matrix and trains only the new rows on Hinglish/Hindi text. Old rows
are protected by a gradient mask, so the base model's behaviour is preserved.

Why only the embedding matrix
-----------------------------
The new tokens change the input distribution. The fastest way to make them
usable is to learn their vectors, not to re-train the whole 2B model. This is
the same recipe that recovered the v2/v3 injected rows, scaled to Qwen3.5.

Usage
-----
  venv/bin/python scripts/warmup_v4_embeddings.py \
    --model models/bharat-v4-tokenized \
    --corpus "/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl" \
    --out models/bharat-v4-warm \
    --steps 500 --batch 2 --seq 512 --lr 1e-3

  # quick smoke test
  venv/bin/python scripts/warmup_v4_embeddings.py --steps 20 --max-docs 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

LN2 = math.log(2)
DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kaise", "nahi", "bhi", "yeh", "toh", "chahiye",
    "bilkul", "bahut", "aap", "main", "hum", "kar", "karo", "hoga", "raha",
    "bhai", "yaar", "kuch", "sab", "thik", "theek", "acha", "accha", "haan",
}


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model
    return model


def load_model(model_id: str, dtype: str):
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
                model_id, dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  {loader_name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise RuntimeError(f"could not load {model_id}") from last_exc


def iter_docs(path: Path, max_docs: int):
    with path.open() as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages")
            if msgs and msgs and msgs[0].get("role") == "system":
                msgs = msgs[1:]
            text = "\n".join(m.get("content", "") for m in msgs) if msgs else obj.get("text", "")
            if text.strip():
                yield text
                n += 1
                if max_docs and n >= max_docs:
                    break


def useful(text: str) -> bool:
    """Cheap pre-filter: Hinglish markers or Devanagari."""
    if DEVA_RE.search(text):
        return True
    words = set(w.lower() for w in re.findall(r"[A-Za-z]+", text))
    return len(words & HINGLISH_MARKERS) >= 2


def build_chunks(docs, tok, seq_len: int, max_chunks: int, old_vocab: int):
    """Pack docs into fixed-length token chunks that actually use new tokens."""
    chunks: list[list[int]] = []
    buf: list[int] = []
    eos = tok.eos_token_id
    docs_used = 0
    for text in docs:
        if not useful(text):
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            continue
        # Only keep text that exercises at least one new token.
        if not any(i >= old_vocab for i in ids):
            continue
        docs_used += 1
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(chunks) >= max_chunks:
                return chunks, docs_used
    if len(buf) >= seq_len // 2:
        buf = buf + [eos] * (seq_len - len(buf))
        chunks.append(buf)
    return chunks, docs_used


def load_windows(path: Path, n: int, chars: int) -> list[str]:
    windows = []
    with path.open() as f:
        for line in f:
            if len(windows) >= n:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                msgs = msgs[1:]
            text = "\n".join(m.get("content", "") for m in msgs)
            if len(text) < chars // 2:
                continue
            windows.append(text[:chars])
    return windows


@torch.no_grad()
def evaluate(model, tok, windows: list[str], old_vocab: int, seq: int, device) -> dict:
    model.eval()
    total_nats = 0.0
    total_chars = 0
    total_tokens = 0
    new_nats = 0.0
    new_tokens = 0
    for text in windows:
        ids = tok.encode(text, add_special_tokens=False)
        ids = ids[:seq]
        if len(ids) < 2:
            continue
        text = tok.decode(ids)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = x[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
        )
        total_nats += float(loss.sum().item())
        total_tokens += int(loss.numel())
        total_chars += len(text)
        mask = shift_labels.reshape(-1) >= old_vocab
        if mask.any():
            new_nats += float(loss[mask].sum().item())
            new_tokens += int(mask.sum().item())
    model.train()
    return {
        "windows": len(windows),
        "chars": total_chars,
        "tokens": total_tokens,
        "bits_per_char": round(total_nats / total_chars / LN2, 4) if total_chars else None,
        "ppl": round(math.exp(total_nats / total_tokens), 3) if total_tokens else None,
        "new_token_nll": round(new_nats / new_tokens, 4) if new_tokens else None,
        "new_token_fraction": round(new_tokens / total_tokens, 4) if total_tokens else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/bharat-v4-tokenized")
    ap.add_argument("--corpus", default="/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl")
    ap.add_argument("--eval-corpus", default="data/processed/valid_gold.jsonl")
    ap.add_argument("--out", default="models/bharat-v4-warm")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="keep <= 5e-4; 0.05 broke generation in the first run")
    ap.add_argument("--optim", default="sgd", choices=["sgd", "adamw"],
                    help="SGD+momentum uses ~1 GB less state than AdamW")
    ap.add_argument("--max-docs", type=int, default=50000)
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--old-vocab", type=int, default=0,
                    help="override the pre-injection tokenizer length")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    print(f"Device: {device}", file=sys.stderr)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.dtype).to(device)
    text_model = get_text_model(model)
    emb = text_model.embed_tokens
    report_path = Path(args.model, "injection_report.json")
    old_vocab = args.old_vocab
    if not old_vocab and report_path.exists():
        old_vocab = int(json.loads(report_path.read_text())["old_tokenizer_len"])
    if not old_vocab:
        raise SystemExit("injection_report.json not found; pass --old-vocab")
    print(f"Embedding rows: {emb.weight.shape[0]}, old vocab: {old_vocab}", file=sys.stderr)

    # Freeze everything, unfreeze only the embedding matrix.
    for p in model.parameters():
        p.requires_grad_(False)
    emb.weight.requires_grad_(True)
    lm_head = model.get_output_embeddings()
    tied = lm_head is not None and lm_head.weight is emb.weight
    if lm_head is not None and not tied:
        lm_head.weight.requires_grad_(True)
        print("Untied lm_head: training both matrices", file=sys.stderr)

    windows = load_windows(Path(args.eval_corpus), args.eval_windows, 1024)
    print(f"Eval windows: {len(windows)}", file=sys.stderr)
    before = evaluate(model, tok, windows, old_vocab, args.seq, device)
    print(f"before: {json.dumps(before)}", file=sys.stderr)

    docs = iter_docs(Path(args.corpus), args.max_docs)
    chunks, docs_used = build_chunks(docs, tok, args.seq, args.steps * args.batch, old_vocab)
    print(f"Chunks: {len(chunks)} from {docs_used} docs", file=sys.stderr)
    if not chunks:
        raise SystemExit("no chunks contained a new token; check the injection")

    params = [emb.weight] + ([] if tied or lm_head is None else [lm_head.weight])
    if args.optim == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    except Exception:  # noqa: BLE001
        pass
    model.train()
    t0 = time.time()
    losses = []
    history = []
    step = 0
    while step < args.steps:
        random.shuffle(chunks)
        for i in range(0, len(chunks) - args.batch + 1, args.batch):
            batch = torch.tensor(chunks[i : i + args.batch], dtype=torch.long, device=device)
            out = model(batch, labels=batch)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                emb.weight.grad[:old_vocab].zero_()
                if lm_head is not None and not tied and lm_head.weight.grad is not None:
                    lm_head.weight.grad[:old_vocab].zero_()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step % 10 == 0:
                avg = sum(losses[-10:]) / len(losses[-10:])
                print(f"step {step}/{args.steps} loss {avg:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", file=sys.stderr, flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, tok, windows, old_vocab, args.seq, device)
                history.append({"step": step, **ev})
                print(f"eval @ {step}: {json.dumps(ev)}", file=sys.stderr, flush=True)
            if step >= args.steps:
                break

    after = evaluate(model, tok, windows, old_vocab, args.seq, device)
    print(f"after: {json.dumps(after)}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    model.save_pretrained(out, safe_serialization=True)
    if report_path.exists() and report_path.parent != out:
        (out / "injection_report.json").write_text(report_path.read_text())
    Path(out, "warmup_report.json").write_text(json.dumps({
        "model": args.model,
        "corpus": args.corpus,
        "steps": step,
        "batch": args.batch,
        "seq": args.seq,
        "lr": args.lr,
        "device": str(device),
        "dtype": args.dtype,
        "old_vocab": old_vocab,
        "new_vocab": int(emb.weight.shape[0]),
        "docs_used": docs_used,
        "chunks": len(chunks),
        "before": before,
        "after": after,
        "history": history,
        "final_loss": round(sum(losses[-50:]) / len(losses[-50:]), 4) if losses else None,
    }, indent=2))
    print(f"Saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
