"""
GSM8K-Hindi (grade-school math word problems, chain-of-thought) evaluation.
Compares base Qwen2.5-1.5B vs Brahmi v2 (+LoRA) with identical protocol:
  - 4-shot CoT exemplars from the train split
  - each model's own chat template
  - greedy decoding (temp=0)
  - exact match on the final number

Dataset: bingbangboom/gsm8k-hindi (GSM8K machine-translated to Hindi, `####N` answers)

Usage:
  venv/bin/python3 scripts/bench_gsm_hi.py [--n 100] [--out docs/bench_gsm_hi.json]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-bf16"
V2_MODEL = "/Volumes/KIOXIA 1TB/brahmi-mlx-trained"
V2_ADAPTER = "models/adapters_v2_best"

MAX_TOKENS = 320
NUM_SHOTS = 4

INSTR = (
    "निम्नलिखित गणित की समस्याओं को हल करें। चरण-दर-चरण सोचें और हर उत्तर के अंत में "
    "'उत्तर: <संख्या>' लिखें।\n\n"
)


def compact_answer(ans: str) -> str:
    """Keep only the final numeric line of a GSM8K answer."""
    m = re.search(r"#{3,4}\s*(-?[\d,]+(?:\.\d+)?)", ans)
    final = m.group(1).replace(",", "") if m else ""
    # strip the calculator annotations <<...>> from the reasoning body
    body = re.sub(r"<<[^>]*>>", "", ans.split("####")[0]).strip()
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    keep = lines[-NUM_SHOTS:] if len(lines) > NUM_SHOTS else lines
    return "\n".join(keep) + f"\nउत्तर: {final}" if final else body


def build_prompt(exemplars, question):
    parts = [INSTR]
    for ex in exemplars:
        parts.append(f"प्रश्न: {ex['question']}\n{compact_answer(ex['answer'])}\n")
    parts.append(f"प्रश्न: {question}")
    return "\n".join(parts)


def extract_answer(text):
    matches = re.findall(r"उत्तर[:：=\s]*(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "").rstrip(".")
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def eval_model(name, model_path, adapter_path, test_set, exemplars, n):
    print(f"\n=== Evaluating {name} ===", file=sys.stderr)
    model, tok = (load(model_path, adapter_path=adapter_path) if adapter_path
                  else load(model_path))
    sampler = make_sampler(temp=0.0)

    correct, total_gen_tokens, t_start = 0, 0, time.time()
    results = []
    for i, row in enumerate(test_set.select(range(n))):
        prompt = build_prompt(exemplars, row["question"])
        chat = [{"role": "user", "content": prompt}]
        try:
            rendered = tok.apply_chat_template(chat, tokenize=False,
                                               add_generation_prompt=True)
            out = generate(model, tok, prompt=rendered, max_tokens=MAX_TOKENS,
                           sampler=sampler, verbose=False)
            total_gen_tokens += len(tok.encode(out))
        except Exception as e:
            out = ""
            print(f"  [{i}] generation error: {e}", file=sys.stderr)
        pred = extract_answer(out)
        gm = re.search(r"#{3,4}\s*(-?[\d,]+(?:\.\d+)?)", row["answer"])
        if gm:
            gold = gm.group(1).replace(",", "")
        else:
            tail_nums = re.findall(r"-?[\d][\d,]*(?:\.\d+)?", row["answer"])
            gold = tail_nums[-1].replace(",", "") if tail_nums else None
        try:
            ok = (gold is not None and pred is not None
                  and abs(float(pred) - float(gold)) < 1e-6)
        except ValueError:
            ok = False
        correct += int(ok)
        results.append({"i": i, "pred": pred, "gold": gold, "ok": ok})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}: acc={correct/(i+1)*100:.1f}%", file=sys.stderr)

    elapsed = time.time() - t_start
    del model
    return {
        "model": name,
        "accuracy_pct": round(correct / len(results) * 100, 2),
        "correct": correct,
        "total": len(results),
        "tok_per_sec": round(total_gen_tokens / elapsed, 1),
        "samples": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="docs/bench_gsm_hi.json")
    args = ap.parse_args()

    ds = load_dataset("bingbangboom/gsm8k-hindi")
    train, test = ds["train_main"], ds["test_main"]
    exemplars = [train[i] for i in range(NUM_SHOTS)]

    summary, detail = {}, {}
    for name, path, adapter in [
        ("base_qwen25_15b_bf16", BASE_MODEL, None),
        ("v2_lora", V2_MODEL, V2_ADAPTER),
    ]:
        r = eval_model(name, path, adapter, test, exemplars, args.n)
        summary[name] = {k: r[k] for k in ("accuracy_pct", "correct", "total", "tok_per_sec")}
        detail[name] = r

    out = {"benchmark": "GSM8K-Hindi", "shots": NUM_SHOTS, "decoding": "greedy",
           "max_tokens": MAX_TOKENS, "summary": summary, "detail": detail}
    print(json.dumps(summary, indent=2))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved to {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
