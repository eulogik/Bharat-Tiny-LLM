#!/usr/bin/env python3
"""Corrected cross-tokenizer language-model benchmark: bits per character.

Why this script exists
----------------------
`scripts/bench_ppl_torch.py` shifted labels twice:

    model(chunk[:-1], labels=chunk[1:])

Hugging Face causal-LM heads shift labels internally, so the model was scored
against token i+2 instead of i+1. That is why the v3 paper reports a base
perplexity of 1,703,571. With correct alignment the same model on the same text
scores ~1.9. This script replaces that protocol.

Protocol
--------
* Raw held-out text is cut into fixed-size character windows. Every model sees
  exactly the same characters, so bits/char is comparable across tokenizers.
* Each window is tokenized with the model's own tokenizer, truncated to the
  model context if needed, and scored with one forward pass.
* Loss is computed with `model(input_ids, labels=input_ids)` and multiplied by
  the number of predicted tokens. No manual label shifting.
* Reports tokens/1000 chars, perplexity (within-model only), bits/char,
  bits/byte (UTF-8), and a bootstrap 95% CI over windows.
* Slices by script mix: hinglish / deva / english / mixed.

Usage
-----
  venv/bin/python scripts/bench_bpc.py \
    --corpus data/processed/valid_gold.jsonl \
    --windows 200 --window-chars 1024 --seq 2048 --dtype fp16 \
    --model Qwen/Qwen3-1.7B-Base \
    --model "/Volumes/KIOXIA 1TB/brahmi-v3-merged" \
    --out docs/bench_bpc_v3.json

The GGUF/llama.cpp numbers in docs/bench_ppl_v3.json remain a useful second
protocol, but this file is the one to cite for bits/char.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LN2 = math.log(2)

DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,23}")

HINGLISH_MARKERS = {
    "hai", "hain", "ha", "kya", "kyun", "kyu", "kaise", "kaisa", "kaisi",
    "nahi", "nahin", "na", "bhi", "hi", "toh", "to", "aur", "ya", "par",
    "lekin", "magar", "agar", "jab", "tab", "ab", "abhi", "phir", "fir",
    "kal", "aaj", "main", "mein", "mai", "hum", "aap", "tum", "tu", "woh",
    "vo", "yeh", "ye", "mera", "meri", "mere", "tera", "teri", "tere",
    "hamara", "hamari", "apna", "apni", "kar", "karo", "karna", "karte",
    "karti", "karta", "kiya", "kiye", "hoga", "hogi", "hote", "hoti",
    "hota", "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "chahiye",
    "chaiye", "sakte", "sakta", "sakti", "bilkul", "bahut", "bohot",
    "zyada", "kam", "kuch", "kucch", "sab", "sabhi", "ekdum", "thik",
    "theek", "acha", "accha", "achha", "bura", "haan", "han", "ji", "bhai",
    "yaar", "dost", "matlab", "samajh", "samjho", "bolo", "batao", "bata",
    "dekho", "dekh", "suno", "sun", "chalo", "chal", "ruko", "liye", "liya",
    "dena", "do", "diya", "di", "lena", "lo", "wala", "wali", "wale", "log",
    "logon", "ghar", "kaam", "paisa", "paise", "bharat", "hindustan", "desh",
    "sarkar", "yojana", "kisan", "kheti", "fasal", "bazar", "mandi", "sehat",
    "dawai", "doctor", "siksha", "school", "college", "exam", "result",
    "naukri", "family", "mummy", "papa", "bhaiya", "didi", "beta", "beti",
    "maa", "dadi", "nana", "nani", "chacha", "mausi",
}


def classify(text: str) -> str:
    latin = len(LATIN_RE.findall(text))
    deva = len(DEVA_RE.findall(text))
    alpha = latin + deva
    if alpha == 0:
        return "other"
    words = [w.lower() for w in WORD_RE.findall(text)]
    markers = sum(1 for w in words if w in HINGLISH_MARKERS)
    latin_ratio = latin / alpha
    if latin_ratio >= 0.75 and markers == 0:
        return "english"
    if deva / alpha >= 0.85:
        return "deva"
    if latin_ratio >= 0.55 and markers >= 2:
        return "hinglish"
    return "mixed"


def iter_documents(path: Path, start_line: int = 0, max_lines: int = 0) -> Iterable[str]:
    with path.open() as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if max_lines and i >= start_line + max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages")
            if msgs:
                text = "\n".join(m.get("content", "") for m in msgs if m.get("content"))
            else:
                text = obj.get("text", "")
            if text.strip():
                yield text


def build_windows(docs: Iterable[str], window_chars: int, max_windows: int) -> list[dict]:
    """Pack documents into character windows; keep each doc intact when short."""
    windows: list[dict] = []
    for doc in docs:
        # Split long documents on sentence-ish boundaries so a window is not
        # one giant run-on line with no natural context.
        start = 0
        while start < len(doc):
            chunk = doc[start : start + window_chars]
            if len(chunk.strip()) >= window_chars // 4:
                windows.append({"text": chunk, "slice": classify(chunk)})
            start += window_chars
            if len(windows) >= max_windows:
                return windows
    return windows


def load_tokenizer(model_id: str):
    """Load a tokenizer, tolerating old list-form `extra_special_tokens` configs.

    v3 checkpoints were saved with `extra_special_tokens` as a list. Newer
    transformers expects a dict and raises AttributeError. We patch a copy in a
    temp directory instead of touching the model artifacts.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except AttributeError as exc:
        # Older transformers error text is just "'list' object has no attribute
        # 'keys'", so do not gate on the field name.
        if "extra_special_tokens" not in str(exc) and "'keys'" not in str(exc):
            raise
        import shutil
        import tempfile

        src = Path(model_id)
        cfg_path = src / "tokenizer_config.json"
        if not cfg_path.exists():
            raise
        cfg = json.loads(cfg_path.read_text())
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg["extra_special_tokens"] = {}
        tmp = Path(tempfile.mkdtemp(prefix="tokfix_"))
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                     "merges.txt", "special_tokens_map.json"):
            if (src / name).exists():
                shutil.copy(src / name, tmp / name)
        (tmp / "tokenizer_config.json").write_text(json.dumps(cfg))
        return AutoTokenizer.from_pretrained(str(tmp), trust_remote_code=True)


def load_model(model_id: str, dtype: torch.dtype):
    """Load a text or multimodal causal LM (Qwen3.5 is image-text-to-text)."""
    import transformers

    last_exc = None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            print(f"[{model_id}] loading with {loader_name} ...", file=sys.stderr, flush=True)
            return loader.from_pretrained(
                model_id, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  {loader_name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise RuntimeError(f"could not load {model_id}") from last_exc


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    # auto
    if device.type == "mps":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def score_model(model_id: str, windows: list[dict], args) -> dict:
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(f"[{model_id}] loading on {device} as {dtype} ...", file=sys.stderr, flush=True)
    tok = load_tokenizer(model_id)
    model = load_model(model_id, dtype)
    model.to(device)
    model.eval()

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, w in enumerate(windows):
            text = w["text"]
            ids = tok.encode(text, add_special_tokens=False, truncation=False)
            if len(ids) > args.seq:
                ids = ids[: args.seq]
                text = tok.decode(ids)
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(x, labels=x)
            n_pred = len(ids) - 1
            nats = float(out.loss.item()) * n_pred
            if not math.isfinite(nats):
                print(f"  skip non-finite window {i}", file=sys.stderr)
                continue
            rows.append(
                {
                    "slice": w["slice"],
                    "chars": len(text),
                    "bytes": len(text.encode("utf-8")),
                    "tokens": n_pred,
                    "nats": nats,
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(windows)} windows, {time.time() - t0:.0f}s",
                      file=sys.stderr, flush=True)

    if not rows:
        raise RuntimeError(f"no scorable windows for {model_id}")

    def aggregate(subset: list[dict]) -> dict:
        chars = sum(r["chars"] for r in subset)
        nbytes = sum(r["bytes"] for r in subset)
        toks = sum(r["tokens"] for r in subset)
        nats = sum(r["nats"] for r in subset)
        ppl = math.exp(nats / toks) if toks else None
        return {
            "windows": len(subset),
            "chars": chars,
            "bytes": nbytes,
            "tokens": toks,
            "nats": round(nats, 3),
            "tokens_per_1000_chars": round(toks / chars * 1000, 2),
            "bits_per_char": round(nats / chars / LN2, 4),
            "bits_per_byte": round(nats / nbytes / LN2, 4),
            "ppl": round(ppl, 3),
        }

    result = aggregate(rows)

    # Bootstrap 95% CI over windows, token-weighted.
    rng = np.random.default_rng(1234)
    n = len(rows)
    boots = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        sample = [rows[j] for j in idx]
        nats = sum(r["nats"] for r in sample)
        chars = sum(r["chars"] for r in sample)
        boots.append(nats / chars / LN2)
    result["bits_per_char_ci95"] = [
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    ]

    by_slice = {}
    for name in ("hinglish", "deva", "english", "mixed", "other"):
        subset = [r for r in rows if r["slice"] == name]
        if subset:
            by_slice[name] = aggregate(subset)
    result["slices"] = by_slice

    print(f"[{model_id}] {json.dumps({k: result[k] for k in ('tokens_per_1000_chars','bits_per_char','bits_per_byte','ppl')})}",
          file=sys.stderr, flush=True)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/processed/valid_gold.jsonl")
    ap.add_argument("--start-line", type=int, default=0)
    ap.add_argument("--max-lines", type=int, default=0,
                    help="0 = all lines")
    ap.add_argument("--windows", type=int, default=200,
                    help="maximum number of character windows")
    ap.add_argument("--window-chars", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=2048,
                    help="token truncation per window")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="docs/bench_bpc.json")
    args = ap.parse_args()

    if not args.model:
        args.model = ["Qwen/Qwen3-1.7B-Base"]

    corpus = Path(args.corpus)
    docs = iter_documents(corpus, args.start_line, args.max_lines)
    windows = build_windows(docs, args.window_chars, args.windows)
    if len(windows) < 4:
        raise SystemExit(f"only {len(windows)} windows found in {corpus}")
    total_chars = sum(len(w["text"]) for w in windows)
    print(f"Corpus {corpus}: {len(windows)} windows, {total_chars:,} chars", file=sys.stderr)
    for name in ("hinglish", "deva", "english", "mixed", "other"):
        c = sum(1 for w in windows if w["slice"] == name)
        if c:
            print(f"  {name}: {c} windows", file=sys.stderr)

    results = {
        "protocol": "fixed character windows; one forward pass per window; "
                    "loss = model(ids, labels=ids); bits/char and bits/byte",
        "script": "scripts/bench_bpc.py",
        "corpus": str(corpus),
        "start_line": args.start_line,
        "window_chars": args.window_chars,
        "seq": args.seq,
        "windows": len(windows),
        "chars": total_chars,
        "slices": {name: sum(1 for w in windows if w["slice"] == name)
                   for name in ("hinglish", "deva", "english", "mixed", "other")},
        "models": {},
    }

    for model_id in args.model:
        results["models"][model_id] = score_model(model_id, windows, args)

    # Pairwise reductions vs the first model.
    names = list(results["models"])
    if len(names) > 1:
        base = results["models"][names[0]]["bits_per_char"]
        for name in names[1:]:
            cur = results["models"][name]["bits_per_char"]
            results["models"][name]["bits_per_char_reduction_vs_first_pct"] = round(
                (1 - cur / base) * 100, 2
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    # Markdown table
    md = [
        "# Bits-per-character benchmark",
        "",
        f"- Corpus: `{corpus}`",
        f"- Windows: {len(windows)} x {args.window_chars} chars ({total_chars:,} chars)",
        f"- Protocol: one forward pass per window, correct causal alignment, "
        f"bits/char + bits/byte",
        "",
        "| Model | tok/1000 chars | bpc | 95% CI | bits/byte | PPL |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for name in names:
        r = results["models"][name]
        ci = r["bits_per_char_ci95"]
        md.append(
            f"| `{name}` | {r['tokens_per_1000_chars']} | {r['bits_per_char']} | "
            f"{ci[0]}–{ci[1]} | {r['bits_per_byte']} | {r['ppl']:,} |"
        )
    md.append("")
    out.with_suffix(".md").write_text("\n".join(md))
    print(json.dumps({k: v for k, v in results.items() if k != "models"}, indent=2))
    print(f"Saved {out} and {out.with_suffix('.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
