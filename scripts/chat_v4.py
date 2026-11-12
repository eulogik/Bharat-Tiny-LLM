#!/usr/bin/env python3
"""Chat with a v4 HF checkpoint (pre-GGUF sanity check).

Loads a checkpoint that uses the injected tokenizer (e.g.
`models/bharat-v4-warm`) and generates replies with the same system prompt
used in training and in BharatBench-Voice.

Usage
-----
  .venv-v4/bin/python scripts/chat_v4.py \
    --model models/bharat-v4-warm \
    --prompt "Bhai, mere tamatar ke patte peele ho rahe hain, kya karu?"

  # interactive
  .venv-v4/bin/python scripts/chat_v4.py --model models/bharat-v4-warm
"""

from __future__ import annotations

import argparse
import sys

import torch

SYSTEM_PROMPT = (
    "You are Bharat-Tiny-LLM, a helpful offline assistant for India. "
    "Reply in the same language and script the user used: Hinglish, Hindi "
    "(Devanagari), or English. Be concise and practical. Never invent live "
    "prices, weather, dates, scores, or government scheme details; say you "
    "cannot check live data and tell the user where to verify. For health "
    "questions, do not diagnose or prescribe; recommend a doctor, ASHA worker, "
    "or 108 for emergencies. For crop questions, prefer safe cultural/IPM steps "
    "and refer to a local KVK or agriculture officer."
)


def load_model(model_id: str, dtype: str):
    import transformers

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            return loader.from_pretrained(
                model_id, dtype=torch_dtype, trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{loader_name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(f"could not load {model_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/bharat-v4-warm")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.dtype).to(args.device)
    model.eval()

    def answer(text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new = out[0, ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip()

    if args.prompt:
        print(answer(args.prompt))
        return

    print("v4 chat — Ctrl-C to exit.", file=sys.stderr)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        print("Bharat:", answer(q))


if __name__ == "__main__":
    main()
