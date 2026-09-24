"""GSM8K-Hindi bench for v3.4 (torch, same protocol as bench_gsm_hi.py)."""
import argparse, json, re, time
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTR = (
    "निम्नलिखित गणित की समस्याओं को हल करें। चरण-दर-चरण सोचें और हर उत्तर के अंत में "
    "'उत्तर: <संख्या>' लिखें।\n\n"
)
NUM_SHOTS = 4

def compact_answer(ans: str) -> str:
    m = re.search(r"#{3,4}\s*(-?[\d,]+(?:\.\d+)?)", ans)
    final = m.group(1).replace(",", "") if m else ""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/k34_export_v34")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="docs/bench_gsm_hi_v34.json")
    args = ap.parse_args()

    ds = load_dataset("bingbangboom/gsm8k-hindi")
    train, test = ds["train_main"], ds["test_main"]
    exemplars = [train[i] for i in range(NUM_SHOTS)]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")

    correct, t0, results = 0, time.time(), []
    for i, row in enumerate(test.select(range(args.n))):
        prompt = build_prompt(exemplars, row["question"])
        chat = [{"role": "user", "content": prompt}]
        try:
            rendered = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tok(rendered, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            text = ""
            print(f"  [{i}] gen error: {e}", flush=True)
        pred = extract_answer(text)
        gm = re.search(r"#{3,4}\s*(-?[\d,]+(?:\.\d+)?)", row["answer"])
        if gm:
            gold = gm.group(1).replace(",", "")
        else:
            tail = re.findall(r"-?[\d][\d,]*(?:\.\d+)?", row["answer"])
            gold = tail[-1].replace(",", "") if tail else None
        try:
            ok = gold is not None and pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            ok = False
        correct += int(ok)
        results.append({"i": i, "pred": pred, "gold": gold, "ok": ok})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n}: acc={correct/(i+1)*100:.1f}%", flush=True)

    elapsed = time.time() - t0
    summary = {
        "model": "v3.4",
        "accuracy_pct": round(correct / args.n * 100, 2),
        "correct": correct,
        "total": args.n,
        "time_sec": round(elapsed, 1),
        "protocol": "4-shot CoT, greedy, exact match",
        "vs_v33": "v3.3=24.0% on n=100",
    }
    print(json.dumps(summary, indent=2))
    out = {"benchmark": "GSM8K-Hindi", "shots": NUM_SHOTS, "decoding": "greedy",
           "max_tokens": args.max_tokens, "summary": summary, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
