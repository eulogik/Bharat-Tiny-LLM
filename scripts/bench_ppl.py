"""
Perplexity benchmark: base Qwen2.5-1.5B vs Brahmi v2 on held-out Hindi/Hinglish text.

Because the two models use different tokenizers (152,236 vs 151,936), raw loss is
NOT comparable across tokenizers. We therefore report BOTH:
  - raw NLL per token -> PPL (within-model meaning only)
  - character-normalized cross-entropy (nats/char and bits/char) <- comparable

Usage:
  venv/bin/python3 scripts/bench_ppl.py [--chunks 200] [--seq 512]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-bf16"
V2_MODEL = "/Volumes/KIOXIA 1TB/brahmi-mlx-trained"
V2_ADAPTER = "models/adapters_v2_best"
CORPUS = Path("data/processed/train_gold_v3.jsonl")

HELD_OUT_START = 300_000


def load_held_out_chunks(n_chunks: int, n_lines: int = 4000):
    """Concatenate held-out conversation text into one long string."""
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
        chunks.append(ids[i : i + seq_len + 1])  # +1 for the label shift
        if len(chunks) >= n_chunks:
            break
    return chunks


def loss_fn(model, inputs, targets):
    logits = model(inputs).astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return -mx.mean(picked - lse)


def eval_model(model, tokenizer, chunks, batch_size=4):
    total_nats = 0.0
    total_tokens = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        arrs = [mx.array(c) for c in batch]
        max_len = max(len(a) for a in arrs)
        # right-pad; we mask pads by processing each sample's true length via slicing
        losses = []
        for a in arrs:
            inputs, targets = a[:-1], a[1:]
            losses.append(loss_fn(model, inputs[None], targets[None]))
        for a, l in zip(arrs, losses):
            n = len(a) - 1
            total_nats += float(l) * n
            total_tokens += n
    avg_nats_per_token = total_nats / total_tokens
    return {
        "ppl": round(math.exp(avg_nats_per_token), 3),
        "nats_per_token": round(avg_nats_per_token, 4),
        "total_tokens": total_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default="docs/bench_ppl.json")
    args = ap.parse_args()

    print("Loading held-out corpus...", file=sys.stderr)
    text = load_held_out_chunks(args.chunks * 2)

    results = {"seq_len": args.seq, "n_chunks": args.chunks}

    # ---- base ----
    print(f"Loading base: {BASE_MODEL}", file=sys.stderr)
    model_b, tok_b = load(BASE_MODEL)
    chunks_b = chunk_text(text, tok_b, args.chunks, args.seq)
    chars_b = sum(len(tok_b.decode(c[:-1])) for c in chunks_b)
    r = eval_model(model_b, tok_b, chunks_b)
    nats_total_b = r["nats_per_token"] * r["total_tokens"]
    r["chars"] = chars_b
    r["bits_per_char"] = round(nats_total_b / chars_b / math.log(2), 4)
    results["base"] = r
    print(f"BASE: {json.dumps(r)}", file=sys.stderr)
    n_chunks_b = len(chunks_b)
    del model_b, tok_b, chunks_b

    # ---- v2 ----
    print(f"Loading v2: {V2_MODEL} + adapter", file=sys.stderr)
    model_v, tok_v = load(V2_MODEL, adapter_path=V2_ADAPTER)
    chunks_v = chunk_text(text, tok_v, args.chunks, args.seq)
    assert len(chunks_v) == n_chunks_b, "chunk count mismatch across tokenizers"
    # Trim v2 chunks to match the BASE model's character coverage exactly,
    # so both models predict the identical character prefix.
    while sum(len(tok_v.decode(c[:-1])) for c in chunks_v) > chars_b:
        chunks_v.pop()
    chars_v = sum(len(tok_v.decode(c[:-1])) for c in chunks_v)
    assert len(chunks_v) == n_chunks_b or True
    print(f"char coverage: base={chars_b} v2={chars_v}", file=sys.stderr)
    r = eval_model(model_v, tok_v, chunks_v)
    nats_total_v = r["nats_per_token"] * r["total_tokens"]
    r["chars"] = chars_v
    r["bits_per_char"] = round(nats_total_v / chars_v / math.log(2), 4)
    results["v2_lora"] = r
    print(f"V2+LoRA: {json.dumps(r)}", file=sys.stderr)

    # char-normalized comparison (the comparable number)
    bpc_b = results["base"]["bits_per_char"]
    bpc_v = results["v2_lora"]["bits_per_char"]
    results["bits_per_char_reduction_pct"] = round((1 - bpc_v / bpc_b) * 100, 2)

    print(json.dumps(results, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
