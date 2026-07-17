"""
Merge 2 trusted HF datasets with gold set.

Sources:
  - NebulaByte/alpaca-gpt4-hindi-hinglish (50K, Alpaca-style bilingual pairs)
  - smangrul/hinglish_self_instruct_v0 (1K, already messages format)

Output: data/processed/train_gold_v2.jsonl
Gold set untouched.
"""
import json, os
from collections import Counter

import pandas as pd

STAGING = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/staging"
GOLD = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/processed/train_gold.jsonl"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/processed/train_gold_v2.jsonl"
PER_SOURCE_CAP = 50000


def clean_content(t):
    if not isinstance(t, str):
        return ""
    t = t.strip()
    return t


def contaminated(t):
    for ch in t:
        cp = ord(ch)
        for lo, hi in [(0x0980, 0x09FF), (0x0C00, 0x0C7F), (0x0B80, 0x0BFF),
                        (0x0400, 0x04FF), (0x4E00, 0x9FFF), (0x0600, 0x06FF),
                        (0x0E00, 0x0E7F), (0xAC00, 0xD7AF), (0x0C80, 0x0CDF),
                        (0x0D00, 0x0D7F), (0x0A00, 0x0A7F)]:
            if lo <= cp <= hi:
                return True
    return False


def load_nebulabyte():
    path = os.path.join(STAGING, "NebulaByte__alpaca-gpt4-hindi-hinglish",
                        "data", "train-00000-of-00001-d18143b18ee5a0f5.parquet")
    df = pd.read_parquet(path)
    out = []
    count = 0
    for _, r in df.iterrows():
        if count >= PER_SOURCE_CAP:
            break
        inp = clean_content(r.get("input", ""))
        outp = clean_content(r.get("output", ""))
        inp_h = clean_content(r.get("input_hinglish", ""))
        outp_h = clean_content(r.get("output_hinglish", ""))
        # Devanagari pair
        if inp and outp and not contaminated(inp) and not contaminated(outp):
            out.append({"messages": [
                {"role": "user", "content": inp},
                {"role": "assistant", "content": outp}
            ]})
            count += 1
            if count >= PER_SOURCE_CAP:
                break
        # Romanized pair
        if inp_h and outp_h:
            out.append({"messages": [
                {"role": "user", "content": inp_h},
                {"role": "assistant", "content": outp_h}
            ]})
    return out


def load_smangrul():
    path = os.path.join(STAGING, "smangrul__hinglish_self_instruct_v0",
                        "data", "train-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    out = []
    count = 0
    for _, r in df.iterrows():
        if count >= PER_SOURCE_CAP:
            break
        msgs_raw = r.get("messages", [])
        if hasattr(msgs_raw, "tolist"):
            msgs = msgs_raw.tolist()
        else:
            msgs = list(msgs_raw) if hasattr(msgs_raw, "__iter__") else [msgs_raw]
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        cleaned = []
        for m in msgs:
            content = clean_content(m.get("content", ""))
            role = m.get("role", "")
            if content and role:
                cleaned.append({"role": role, "content": content})
        if len(cleaned) >= 2:
            out.append({"messages": cleaned})
            count += 1
    return out


def main():
    samples = []
    per_src = Counter()

    neb = load_nebulabyte()
    samples.extend(neb)
    per_src["nebulabyte"] = len(neb)
    print(f"NebulaByte: {len(neb)}")

    sman = load_smangrul()
    samples.extend(sman)
    per_src["smangrul"] = len(sman)
    print(f"smangrul: {len(sman)}")

    print(f"\nTotal staged: {len(samples)}")
    for k, v in per_src.most_common():
        print(f"  {k}: {v}")

    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    print(f"Gold train: {len(gold)}")

    final = list(gold)
    prompt_seen = Counter()
    for m in gold:
        for msg in m["messages"]:
            if msg["role"] == "user":
                prompt_seen[msg["content"]] += 1

    dup = 0
    for s in samples:
        p = s["messages"][0]["content"]
        if prompt_seen[p] >= 10:
            dup += 1
            continue
        prompt_seen[p] += 1
        final.append(s)

    print(f"Merged: {len(final)} (staged {len(samples)}, dup-capped {dup})")

    with open(OUT, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
