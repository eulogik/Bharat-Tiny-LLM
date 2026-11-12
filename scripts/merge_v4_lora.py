#!/usr/bin/env python3
"""Merge a v4 LoRA adapter into the warmed Qwen3.5-2B checkpoint.

Usage
-----
  python scripts/merge_v4_lora.py \
    --base /kaggle/working/bharat-v4-warm \
    --adapter /kaggle/working/bharat-v4-lora \
    --out /kaggle/working/bharat-v4-merged

Then convert to GGUF with llama.cpp:
  python convert_hf_to_gguf.py /kaggle/working/bharat-v4-merged --outtype f16 \
    --outfile bharat-v4-f16.gguf
  llama-quantize bharat-v4-f16.gguf bharat-v4-q4_k_m.gguf Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


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
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    base = load_base(args.base, args.dtype)
    print(f"Loading adapter {args.adapter} ...", file=sys.stderr)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging ...", file=sys.stderr)
    model = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    # Prefer the adapter's tokenizer (it carries the injected vocab).
    tok = None
    for candidate in (args.adapter, args.base):
        try:
            tok = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            break
        except Exception:  # noqa: BLE001
            continue
    if tok is None:
        raise SystemExit("could not load a tokenizer")
    tok.save_pretrained(out)

    # save_pretrained drops the MTP/NextN draft head and some multimodal
    # sidecar files. Copy the small non-weight files so the checkpoint is a
    # complete HF repo; convert the GGUF with --no-mtp (see report below).
    base_dir = Path(args.base)
    copied = []
    for name in (
        "preprocessor_config.json", "video_preprocessor_config.json",
        "chat_template.jinja", "vocab.json", "merges.txt", "LICENSE",
        "generation_config.json",
    ):
        src = base_dir / name
        dst = out / name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied.append(name)

    report = {
        "base": args.base,
        "adapter": args.adapter,
        "out": str(out),
        "vocab_size": len(tok),
        "dtype": args.dtype,
        "copied_sidecars": copied,
        "mtp_head": "not present after save_pretrained; convert with --no-mtp",
    }
    (out / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
