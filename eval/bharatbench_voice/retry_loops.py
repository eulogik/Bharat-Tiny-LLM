#!/usr/bin/env python3
"""Regenerate degenerate (looping) answers in a BharatBench-Voice run.

The benchmark runner now retries a looping answer once with
``repeat_penalty=1.15``. This script applies the same rule to a run recorded
before that guard existed, so the stored run matches the shipped system. It only
touches rows with a 5-gram loop ratio >= ``--threshold``.

Usage
-----
  python eval/bharatbench_voice/retry_loops.py \
    --run eval/bharatbench_voice/runs/system_full.jsonl \
    --out /tmp/system_full_retry.jsonl \
    --server http://127.0.0.1:8096/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))

from text_quality import is_degenerate, loop_ratio  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8096/v1")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.run).read_text().splitlines() if line.strip()]
    changed = []
    for r in rows:
        if r.get("error") or r.get("domain") == "health":
            continue
        text = r.get("response", "")
        ratio = loop_ratio(text)
        if not is_degenerate(text, threshold=args.threshold):
            continue
        before = ratio
        payload = {
            "messages": [
                {"role": "system", "content": r.get("system_prompt", "")},
                {"role": "user", "content": r["text"]},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "repeat_penalty": 1.15,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        resp = requests.post(args.server.rstrip("/") + "/chat/completions", json=payload,
                             timeout=args.timeout)
        resp.raise_for_status()
        new_text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        r["response"] = new_text
        r["loop_retry"] = True
        r["loop_ratio_before"] = round(before, 3)
        r["loop_ratio"] = round(loop_ratio(new_text), 3)
        r["total_s"] = round(time.perf_counter() - t0, 3)
        changed.append((r["id"], round(before, 3), r["loop_ratio"]))

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(json.dumps({"retried": len(changed), "rows": changed}, indent=2))
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
