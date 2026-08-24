"""
MGSM-Hindi (grade-school math, chain-of-thought) evaluation.
Compares base Qwen2.5-1.5B vs Brahmi v2 (+LoRA) with identical protocol:
  - 4-shot CoT exemplars from the train split
  - each model's own chat template
  - greedy decoding (temp=0)
  - exact match on the final number

Usage:
  venv/bin/python3 scripts/bench_mgsm.py [--n 100] [--out docs/bench_mgsm.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-bf16"
V2_MODEL = "/Volumes/KIOXIA 1TB/brahmi-mlx-trained"
V2_ADAPTER = "/Volumes/KIOXIA 1TB/mlx_lora_adapter/0000400_adapters.safetensors"

MAX_TOKENS = 320
NUM_SHOTS = 4


def build_prompt(exemplars, question):
    parts = [
        "निम्नलिखित गणित की समस्याओं को हल करें। हर उत्तर के अंत में 'उत्तर: <संख्या>' लिखें।\n"
    ]
    for ex in exemplars:
        parts.append(f"प्रश्न: {ex['question']}\n{ex['answer'].strip()}\n")
    parts.append(f"प्रश्न: {question}")
    return "\n".join(parts)


def extract_answer(text):
    # prefer explicit marker, else last number in the text
    matches = re.findall(r"उत्तर[:：]\s*(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"-?[\d][\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def eval_model(name, model_path, adapter_path, test_set, exemplars, n):
    print(f"\n=== Evaluating {name} ===", file=sys.stderr)
    model, tok = load(model_path, adapter_path=adapter_path) if adapter_path else load(model_path)
    sampler = make_sampler(temp=0.0)

    correct = 0
    results = []
    for i, row in enumerate(test_set.select(range(n))):
        prompt = build_prompt(exemplars, row["question"])
        chat = [{"role": "user", "content": prompt}]
        try:
            rendered = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            out = generate(model, tok, prompt=rendered, max_tokens=MAX_TOKENS,
                           sampler=sampler, verbose=False)
        except Exception as e:
            out = ""
            print(f"  [{i}] generation error: {e}", file=sys.stderr)
        pred = extract_answer(out)
        gold = str(row["answer_number"])
        ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        correct += int(ok)
        results.append({"i": i, "pred": pred, "gold": gold, "ok": ok})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}: acc={correct/(i+1)*100:.1f}%", file=sys.stderr)

    del model
    return {
        "model": name,
        "accuracy_pct": round(correct / len(results) * 100, 2),
        "correct": correct,
        "total": len(results),
        "samples": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="docs/bench_mgsm.json")
    args = ap.parse_args()

    ds = load_dataset("juletxara/mgsm", "hi")
    train, test = ds["train"], ds["test"]
    exemplars = [train[i] for i in range(NUM_SHOTS)]
    # compact the exemplar answers to their final line to keep prompts short
    for ex in exemplars:
        lines = [l for l in ex["answer"].strip().splitlines() if l.strip()]
        ex["answer"] = lines[-1] if lines else ex["answer"]

    summary = {}
    detail = {}

    for name, path, adapter in [
        ("base_qwen25_15b_bf16", BASE_MODEL, None),
        ("v2_lora", V2_MODEL, V2_ADAPTER),
    ]:
        r = eval_model(name, path, adapter, test, exemplars, args.n)
        summary[name] = {"accuracy_pct": r["accuracy_pct"], "correct": r["correct"], "total": r["total"]}
        detail[name] = r

    out = {"summary": summary, "shots": NUM_SHOTS, "decoding": "greedy", "detail": detail}
    print(json.dumps(summary, indent=2))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved to {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
