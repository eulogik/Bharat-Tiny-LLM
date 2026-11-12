#!/usr/bin/env python3
"""Inject Hinglish tokens into Qwen3.5-2B and mean-initialize the new rows.

This is the v4 replacement for the Devanagari-only `scripts/inject_vocab.py`.
The token list comes from `scripts/mine_hinglish_tokens.py`.

What it does
------------
1. Tokenizes every new token with the *old* tokenizer to get its constituent
   pieces (this must happen before `add_tokens`, otherwise the tokenizer
   returns the new token itself).
2. Adds the tokens to the tokenizer.
3. Loads Qwen3.5-2B, resizes the (tied) embedding table, and initializes every
   new row to the mean of its constituent old embeddings.
4. Saves the tokenizer + model + an injection report.

Why mean init
-------------
A random new row is noise for the first N steps. Mean init gives the model a
sensible starting point: " hain" starts as the average of " h" + "ain", so the
next-token distribution is close to the base model before warmup.

Usage
-----
  # dry run: compute constituents and validate round-trip, no model download
  venv/bin/python scripts/inject_v4_tokens.py \
    --tokens data/v4/hinglish_tokens.json --dry-run

  # write the injected checkpoint
  venv/bin/python scripts/inject_v4_tokens.py \
    --model Qwen/Qwen3.5-2B \
    --tokens data/v4/hinglish_tokens.json \
    --out models/bharat-v4-tokenized

  # optional: only inject whole words (drops partial Devanagari n-grams)
  venv/bin/python scripts/inject_v4_tokens.py --whole-words-only ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

DEVA_LETTER = re.compile(r"^[\u0904-\u0939\u0958-\u095F\u0960-\u0963\u0971-\u097F]")
LATIN_LETTER = re.compile(r"^[A-Za-z]")


def load_tokens(path: Path, whole_words_only: bool, limit: int = 0) -> list[str]:
    data = json.loads(path.read_text())
    tokens = data["tokens"] if isinstance(data, dict) else data
    if whole_words_only:
        # A token is a whole word if, after the leading space, it starts with a
        # letter. This drops n-grams that begin with a combining mark (" ्टे")
        # or end mid-word (" ेकल").
        tokens = [t for t in tokens if DEVA_LETTER.match(t.lstrip()) or LATIN_LETTER.match(t.lstrip())]
    if limit:
        tokens = tokens[:limit]
    # De-duplicate while preserving order.
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_text_model(model):
    """Return the language-model submodule for Qwen3.5 / Qwen3 / Llama."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model
    if hasattr(model, "embed_tokens"):
        return model
    raise AttributeError("could not locate embed_tokens on this model")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tokens", default="data/v4/hinglish_tokens.json")
    ap.add_argument("--out", default="models/bharat-v4-tokenized")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--whole-words-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--roundtrip-text", default="data/processed/valid_gold.jsonl",
                    help="sample text for decode round-trip validation")
    ap.add_argument("--roundtrip-lines", type=int, default=200)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokens = load_tokens(Path(args.tokens), args.whole_words_only, args.limit)
    if not tokens:
        raise SystemExit("no tokens selected")
    print(f"Selected {len(tokens)} tokens", file=sys.stderr)

    print(f"Loading old tokenizer {args.model} ...", file=sys.stderr)
    old_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    old_len = len(old_tok)

    constituents: dict[str, list[int]] = {}
    pieces: dict[str, list[str]] = {}
    failures = []
    for t in tokens:
        ids = old_tok.encode(t, add_special_tokens=False)
        if not ids:
            failures.append(t)
            continue
        constituents[t] = ids
        pieces[t] = old_tok.convert_ids_to_tokens(ids)

    print(f"Constituents computed: {len(constituents)} ok, {len(failures)} failures", file=sys.stderr)

    # Round-trip check: added tokens must not change the decoded text.
    new_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    added = new_tok.add_tokens(tokens, special_tokens=False)
    new_len = len(new_tok)
    roundtrip_ok = True
    roundtrip_examples = []
    rt_path = Path(args.roundtrip_text)
    if rt_path.exists():
        checked = 0
        with rt_path.open() as f:
            for line in f:
                if checked >= args.roundtrip_lines:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = "\n".join(m.get("content", "") for m in obj.get("messages", []))
                if not text:
                    continue
                enc = new_tok.encode(text, add_special_tokens=False)
                dec = new_tok.decode(enc)
                if dec != text:
                    roundtrip_ok = False
                    roundtrip_examples.append({"text": text[:120], "decoded": dec[:120]})
                checked += 1
        print(f"Round-trip checked {checked} docs: {'OK' if roundtrip_ok else 'MISMATCH'}",
              file=sys.stderr)

    report = {
        "model": args.model,
        "tokens_file": str(args.tokens),
        "whole_words_only": args.whole_words_only,
        "tokens_requested": len(tokens),
        "tokens_added": added,
        "old_tokenizer_len": old_len,
        "new_tokenizer_len": new_len,
        "roundtrip_ok": roundtrip_ok,
        "roundtrip_examples": roundtrip_examples[:5],
        "failures": failures,
        "constituents": {t: pieces[t] for t in tokens if t in pieces},
    }

    if args.dry_run:
        print(json.dumps({k: v for k, v in report.items() if k != "constituents"}, indent=2))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "injection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Dry run: wrote {out/'injection_report.json'}", file=sys.stderr)
        return

    model = load_model(args.model, args.dtype)
    text_model = get_text_model(model)
    old_model_vocab = text_model.embed_tokens.weight.shape[0]
    print(f"Model embedding rows before resize: {old_model_vocab}", file=sys.stderr)

    model.resize_token_embeddings(new_len)
    text_model = get_text_model(model)
    emb = text_model.embed_tokens.weight
    print(f"Model embedding rows after resize: {emb.shape[0]}", file=sys.stderr)
    if emb.shape[0] < new_len:
        raise RuntimeError(f"resize produced {emb.shape[0]} rows, need {new_len}")

    # Mean-init every new token row. New token id = old_len + i, which may
    # overlap the base model's unused padded rows; overwrite them too.
    init_ok = 0
    with torch.no_grad():
        for i, t in enumerate(tokens):
            ids = constituents.get(t)
            if not ids:
                continue
            target = old_len + i
            if target >= emb.shape[0]:
                break
            vec = emb[ids].to(torch.float32).mean(dim=0).to(emb.dtype)
            emb[target] = vec
            init_ok += 1

    report.update({
        "old_model_vocab": old_model_vocab,
        "new_model_vocab": int(emb.shape[0]),
        "rows_mean_initialized": init_ok,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
    })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving tokenizer + model to {out} ...", file=sys.stderr)
    new_tok.save_pretrained(out)
    model.save_pretrained(out, safe_serialization=True)
    (out / "injection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "constituents"}, indent=2))
    print(f"Saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
