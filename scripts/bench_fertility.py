"""
Fertility / compression benchmark: base Qwen2.5 tokenizer vs Brahmi-injected v2.

Metrics (computed on held-out text never used for token selection):
  - tokens per 1000 characters
  - characters per token (CPT)
  - token savings %
Computed separately on:
  - Devanagari runs only (script-specific compression)
  - full mixed Hinglish+Devanagari conversations (end-to-end effect)

Usage:
  venv/bin/python3 scripts/bench_fertility.py [--n 5000] [--out docs/bench_fertility.json]
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

BASE_TOK = "Qwen/Qwen2.5-1.5B"
V2_MODEL = "/Volumes/KIOXIA 1TB/brahmi-mlx-trained"
CORPUS = Path("data/processed/train_gold_v3.jsonl")

# Lines 0-15K were used for token selection/val during budget analysis.
# Use a distant slice as held-out data.
HELD_OUT_START = 300_000

DEVA_RANGE = (0x0900, 0x097F)

SAMPLE_PHRASES = [
    "नमस्ते, आप कैसे हैं?",
    "ज़रूरी बात है क्या करते हो",
    "हम आपको बताना चाहते हैं",
    "कितने बजे मिलना है?",
    "चाय पी लो, फिर बात करते हैं",
]


def extract_deva_runs(text: str):
    runs, run = [], ""
    for ch in text:
        if DEVA_RANGE[0] <= ord(ch) <= DEVA_RANGE[1]:
            run += ch
        else:
            if len(run) >= 2:
                runs.append(run)
            run = ""
    if len(run) >= 2:
        runs.append(run)
    return runs


def load_held_out(n_lines: int):
    texts = []
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
            texts.append(" ".join(m["content"] for m in msgs))
    return texts


def measure(tok, texts):
    total_tokens = 0
    total_chars = 0
    for t in texts:
        total_chars += len(t)
        total_tokens += len(tok.encode(t))
    return {
        "chars": total_chars,
        "tokens": total_tokens,
        "tokens_per_1000_chars": round(total_tokens / total_chars * 1000, 1),
        "chars_per_token": round(total_chars / total_tokens, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="held-out conversation lines")
    ap.add_argument("--out", default="docs/bench_fertility.json")
    args = ap.parse_args()

    print(f"Loading held-out slice ({args.n} lines from line {HELD_OUT_START})...", file=sys.stderr)
    texts = load_held_out(args.n)
    print(f"Loaded {len(texts)} conversations", file=sys.stderr)

    deva_runs = []
    for t in texts:
        deva_runs.extend(extract_deva_runs(t))
    print(f"Devanagari runs: {len(deva_runs)}", file=sys.stderr)

    base_tok = AutoTokenizer.from_pretrained(BASE_TOK)
    v2_tok = AutoTokenizer.from_pretrained(V2_MODEL)
    assert len(v2_tok) == len(base_tok) + 300, (
        f"expected +300 vocab, got {len(v2_tok)} vs {len(base_tok)}"
    )

    results = {"base_vocab": len(base_tok), "v2_vocab": len(v2_tok), "n_convs": len(texts)}

    # --- Devanagari-only ---
    b_deva = measure(base_tok, deva_runs)
    v_deva = measure(v2_tok, deva_runs)
    results["devanagari"] = {
        "base": b_deva,
        "v2": v_deva,
        "savings_pct": round((1 - v_deva["tokens"] / b_deva["tokens"]) * 100, 2),
    }

    # --- Full mixed text ---
    b_full = measure(base_tok, texts)
    v_full = measure(v2_tok, texts)
    results["full_text"] = {
        "base": b_full,
        "v2": v_full,
        "savings_pct": round((1 - v_full["tokens"] / b_full["tokens"]) * 100, 2),
    }

    # --- Canonical phrase table ---
    phrases = []
    for p in SAMPLE_PHRASES:
        nb, nv = len(base_tok.encode(p)), len(v2_tok.encode(p))
        phrases.append({"text": p, "base_tokens": nb, "v2_tokens": nv,
                        "savings_pct": round((1 - nv / nb) * 100, 1)})
    results["phrases"] = phrases

    print(json.dumps(results, indent=2, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
