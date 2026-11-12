#!/usr/bin/env python3
"""Summarize a BharatBench-Voice run and write a failure review.

Usage
-----
  python eval/bharatbench_voice/analyze.py \
    --run eval/bharatbench_voice/runs/qwen35_2b_base.jsonl \
    --out eval/bharatbench_voice/runs/qwen35_2b_base.review.md
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def reasons_for(r: dict) -> list[str]:
    c = r.get("checks", {})
    reasons = []
    if not c.get("language_match"):
        reasons.append("language")
    if not c.get("junk_free"):
        reasons.append("junk")
    safety = c.get("safety", {})
    if safety and not safety.get("pass", True):
        reasons.append("safety:" + ",".join(safety.get("flags", [])))
    live = c.get("live_data", {})
    if live.get("applicable") and not live.get("pass"):
        reasons.append("live_data")
    math = c.get("math", {})
    if math.get("applicable") and not math.get("pass"):
        reasons.append("math")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.run).read_text().splitlines() if line.strip()]
    total = len(rows)
    by_domain = collections.defaultdict(lambda: collections.Counter())
    reason_counts = collections.Counter()
    flagged = []

    for r in rows:
        reasons = reasons_for(r)
        d = r.get("domain", "?")
        by_domain[d]["n"] += 1
        if reasons:
            by_domain[d]["flagged"] += 1
            for reason in reasons:
                reason_counts[reason.split(":")[0]] += 1
            flagged.append((r, reasons))

    ttfts = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    totals = [r["total_s"] for r in rows if r.get("total_s")]

    lines = [
        "# BharatBench-Voice failure review",
        "",
        f"- Run: `{args.run}`",
        f"- Prompts: {total}",
        f"- Flagged: {len(flagged)} ({len(flagged)/total*100:.1f}%)",
        f"- TTFT p50/p95: {statistics.median(ttfts):.3f} / "
        f"{sorted(ttfts)[int(len(ttfts)*0.95)-1]:.3f} s" if ttfts else "- TTFT: n/a",
        f"- Total p50/p95: {statistics.median(totals):.3f} / "
        f"{sorted(totals)[int(len(totals)*0.95)-1]:.3f} s" if totals else "- Total: n/a",
        "",
        "## Flags by domain",
        "",
        "| Domain | n | Flagged | Rate |",
        "|---|---:|---:|---:|",
    ]
    for d, c in sorted(by_domain.items()):
        lines.append(f"| {d} | {c['n']} | {c['flagged']} | {c['flagged']/c['n']*100:.1f}% |")
    lines += ["", "## Flag counts", ""]
    for reason, n in reason_counts.most_common():
        lines.append(f"- {reason}: {n}")

    lines += ["", "## Examples", ""]
    shown = 0
    for r, reasons in flagged:
        if shown >= args.examples:
            break
        shown += 1
        lines += [
            f"### {r['id']} — {r.get('domain')} / {r.get('language')} — {', '.join(reasons)}",
            "",
            f"**Prompt:** {r.get('text','')}",
            "",
            f"**Response:** {r.get('response','')[:900]}",
            "",
        ]

    md = "\n".join(lines)
    print(md[:3000])
    if args.out:
        Path(args.out).write_text(md)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
