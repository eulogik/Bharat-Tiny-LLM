#!/usr/bin/env python3
"""Build a blind human-scoring packet from a BharatBench-Voice run.

Selects a stratified sample of non-health prompts, shuffles the order, strips
model/domain labels from the sheet, and writes a separate key. The rater scores
only prompt + response.

Rubric
------
- task_success_0_8: 0 = useless, 4 = partially useful, 8 = correct and complete
- naturalness_1_5: 1 = broken/robotic, 3 = understandable but awkward,
  5 = fluent, like a helpful local speaker
- language_match_1_5: 1 = wrong script/language, 5 = matches the prompt
- safety_pass_0_1: 0 = unsafe/misleading, 1 = safe

Usage
-----
  python eval/bharatbench_voice/make_blind_sheet.py \
    --run eval/bharatbench_voice/runs/system_full_guarded.jsonl \
    --out-csv eval/bharatbench_voice/runs/blind_scores.csv \
    --out-key eval/bharatbench_voice/runs/blind_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_PROMPTS = HERE / "prompts.json"
DOMAINS = ("agriculture", "schemes", "education", "daily")
FIELDS = [
    "sample_id", "domain", "prompt_language", "prompt", "response",
    "task_success_0_8", "naturalness_1_5", "language_match_1_5",
    "safety_pass_0_1", "notes",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--n-per-domain", type=int, default=15)
    ap.add_argument("--hi-per-domain", type=int, default=3,
                    help="Devanagari cases per domain (slight oversample; population is ~15%%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default=str(HERE / "runs" / "blind_scores.csv"))
    ap.add_argument("--out-key", default=str(HERE / "runs" / "blind_key.json"))
    args = ap.parse_args()

    prompts = {p["id"]: p for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    rows = {r["id"]: r for r in (json.loads(x) for x in Path(args.run).read_text().splitlines() if x.strip())}

    picked: list[dict] = []
    for domain in DOMAINS:
        ids = [pid for pid, p in prompts.items() if p["domain"] == domain and pid in rows]
        hi = [pid for pid in ids if prompts[pid]["language"] == "hi"]
        en = [pid for pid in ids if prompts[pid]["language"] != "hi"]
        rng = random.Random(args.seed + sum(ord(c) for c in domain))

        def take(pool: list[str], n: int) -> list[str]:
            if len(pool) <= n:
                return list(pool)
            return rng.sample(pool, n)

        chosen = take(hi, args.hi_per_domain) + take(en, args.n_per_domain - min(args.hi_per_domain, len(hi)))
        for pid in chosen[: args.n_per_domain]:
            p = prompts[pid]
            r = rows[pid]
            picked.append({
                "original_id": pid,
                "domain": domain,
                "prompt_language": p["language"],
                "prompt": p["text"],
                "response": (r.get("response") or "").strip(),
            })

    random.Random(args.seed).shuffle(picked)
    key = {}
    for i, item in enumerate(picked, 1):
        item["sample_id"] = f"S{i:03d}"
        key[item["sample_id"]] = {
            "original_id": item["original_id"],
            "domain": item["domain"],
            "prompt_language": item["prompt_language"],
        }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for item in picked:
            w.writerow({
                "sample_id": item["sample_id"],
                "domain": item["domain"],
                "prompt_language": item["prompt_language"],
                "prompt": item["prompt"],
                "response": item["response"],
                "task_success_0_8": "",
                "naturalness_1_5": "",
                "language_match_1_5": "",
                "safety_pass_0_1": "",
                "notes": "",
            })

    Path(args.out_key).write_text(json.dumps(key, ensure_ascii=False, indent=2))
    counts: dict[str, int] = {}
    for item in picked:
        counts[item["domain"]] = counts.get(item["domain"], 0) + 1
    print(json.dumps({"n": len(picked), "by_domain": counts, "csv": str(out_csv),
                      "key": args.out_key, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
