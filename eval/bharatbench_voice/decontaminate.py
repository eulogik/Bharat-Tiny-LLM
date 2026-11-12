#!/usr/bin/env python3
"""Remove near-duplicate prompts from a BharatBench-Voice run.

The SFT templates deliberately cover crop x symptom, scheme x question and
health x scenario. That is legitimate safety training, but it means some
benchmark prompts share most of their words with a training prompt. Scoring on
the full 200 measures "did the model learn the safety template"; scoring on the
decontaminated subset measures generalization.

This script computes Jaccard similarity between each benchmark prompt and every
SFT user prompt, then keeps only prompts below the threshold.

Usage
-----
  python eval/bharatbench_voice/decontaminate.py \
    --run eval/bharatbench_voice/runs/bharat_v4.jsonl \
    --sft data/v4/sft_curated/train.jsonl \
    --threshold 0.6 \
    --out eval/bharatbench_voice/runs/bharat_v4.decon.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\u0900-\u097F]+", " ", text)
    return set(text.split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_sft(path: Path) -> list[set[str]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in obj.get("messages", []):
                if m.get("role") == "user":
                    out.append(norm(m.get("content", "")))
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sft", default="data/v4/sft_curated/train.jsonl")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sft = load_sft(Path(args.sft))
    rows = [json.loads(line) for line in Path(args.run).read_text().splitlines() if line.strip()]

    kept, dropped = [], []
    for r in rows:
        b = norm(r.get("text", ""))
        best = max((jaccard(b, s) for s in sft), default=0.0)
        r = {**r, "sft_max_jaccard": round(best, 3)}
        if best >= args.threshold:
            dropped.append(r)
        else:
            kept.append(r)

    out = Path(args.out) if args.out else Path(args.run).with_suffix(".decon.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")

    print(json.dumps({
        "run": args.run,
        "threshold": args.threshold,
        "total": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_ids": [r["id"] for r in dropped],
        "kept_ids": [r["id"] for r in kept],
        "out": str(out),
    }, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
