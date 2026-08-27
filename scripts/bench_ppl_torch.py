"""
Perplexity benchmark: Qwen3-1.7B-Base vs Brahmi v3 (injected + SFT) on held-out Hindi/Hinglish text.

Reports:
  - raw NLL per token -> PPL (within-model meaning only)
  - character-normalized cross-entropy (nats/char and bits/char) <- comparable

Usage:
  python3 scripts/bench_ppl_torch.py [--chunks 200] [--seq 512]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
V3_MODEL = "/Volumes/KIOXIA 1TB/brahmi-v3-merged"
CORPUS = Path("data/processed/train_gold_v3.jsonl")
HELD_OUT_START = 300_000


def load_held_out_chunks(n_chunks: int, n_lines: int = 4000):
    parts = []
    with CORPUS.open() as f:
        for i, line in enumerate(f):
            if i < HELD_OUT_START:
                continue
            if i >= HELD_OUT_START + n_lines:
                break
            try:
                msgs = json.loads(line)["messages"]
            except Exception:
                continue
            parts.append(" ".join(m["content"] for m in msgs))
    return "\n".join(parts)


def chunk_text(text: str, tokenizer, n_chunks: int, seq_len: int):
    ids = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        chunks.append(ids[i : i + seq_len + 1])
        if len(chunks) >= n_chunks:
            break
    return chunks


def eval_model(model, tokenizer, chunks, seq_len, device="cpu"):
    total_nats = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            if i % 50 == 0:
                print(f"  chunk {i}/{len(chunks)}...", file=sys.stderr, flush=True)
            ids = torch.tensor([chunk[:-1]], device=device)
            labels = torch.tensor([chunk[1:]], device=device)
            out = model(ids, labels=labels)
            # out.loss is mean NLL over tokens in this sequence
            n = len(chunk) - 1
            total_nats += out.loss.item() * n
            total_tokens += n
    avg_nats = total_nats / total_tokens
    return {
        "ppl": round(math.exp(avg_nats), 3),
        "nats_per_token": round(avg_nats, 4),
        "total_tokens": total_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default="docs/bench_ppl_v3.json")
    args = ap.parse_args()

    print("Loading held-out corpus...", file=sys.stderr)
    text = load_held_out_chunks(args.chunks * 2)

    results = {"seq_len": args.seq, "n_chunks": args.chunks}

    # ---- base ----
    print(f"Loading base: {BASE_MODEL}", file=sys.stderr, flush=True)
    tok_b = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model_b = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, trust_remote_code=True)
    model_b.eval()
    chunks_b = chunk_text(text, tok_b, args.chunks, args.seq)
    chars_b = sum(len(tok_b.decode(c[:-1])) for c in chunks_b)
    print(f"  {len(chunks_b)} chunks, {chars_b} chars", file=sys.stderr, flush=True)
    r = eval_model(model_b, tok_b, chunks_b, args.seq)
    nats_total_b = r["nats_per_token"] * r["total_tokens"]
    r["chars"] = chars_b
    r["bits_per_char"] = round(nats_total_b / chars_b / math.log(2), 4)
    results["base"] = r
    print(f"BASE: {json.dumps(r)}", file=sys.stderr)
    n_chunks_b = len(chunks_b)
    del model_b, tok_b

    # ---- v3 ----
    print(f"Loading v3: {V3_MODEL}", file=sys.stderr, flush=True)
    tok_v = AutoTokenizer.from_pretrained(V3_MODEL, trust_remote_code=True)
    model_v = AutoModelForCausalLM.from_pretrained(V3_MODEL, dtype=torch.float32, trust_remote_code=True)
    model_v.eval()
    chunks_v = chunk_text(text, tok_v, args.chunks, args.seq)
    while sum(len(tok_v.decode(c[:-1])) for c in chunks_v) > chars_b:
        chunks_v.pop()
    chars_v = sum(len(tok_v.decode(c[:-1])) for c in chunks_v)
    print(f"  {len(chunks_v)} chunks, {chars_v} chars", file=sys.stderr, flush=True)
    r = eval_model(model_v, tok_v, chunks_v, args.seq)
    nats_total_v = r["nats_per_token"] * r["total_tokens"]
    r["chars"] = chars_v
    r["bits_per_char"] = round(nats_total_v / chars_v / math.log(2), 4)
    results["v3"] = r
    print(f"V3: {json.dumps(r)}", file=sys.stderr)

    bpc_b = results["base"]["bits_per_char"]
    bpc_v = results["v3"]["bits_per_char"]
    results["bits_per_char_reduction_pct"] = round((1 - bpc_v / bpc_b) * 100, 2)

    print(json.dumps(results, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
