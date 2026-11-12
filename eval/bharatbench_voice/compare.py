#!/usr/bin/env python3
"""Compare two BharatBench-Voice runs and apply the v4 launch gate.

Deterministic metrics only. Task success and naturalness still need blind human
scoring, but this script answers the first question: did v4 fix the language,
safety, live-data and math failures without regressing latency?

Usage
-----
  python eval/bharatbench_voice/compare.py \
    --base eval/bharatbench_voice/runs/qwen35_2b_base_nonstream.jsonl \
    --candidate eval/bharatbench_voice/runs/bharat_v4.jsonl \
    --out eval/bharatbench_voice/runs/v4_gate.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DOMAINS = ("agriculture", "schemes", "education", "health", "daily")
METRICS = (
    ("language_match", "Language match"),
    ("junk_free", "Junk-free"),
    ("safety_pass", "Safety pass"),
    ("live_data_pass", "Live-data honesty"),
    ("math_pass", "Math pass"),
)


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def rate(rows: list[dict], metric: str) -> float | None:
    vals = []
    for r in rows:
        c = r.get("checks", {})
        if metric == "safety_pass":
            v = c.get("safety", {}).get("pass")
        elif metric == "live_data_pass":
            v = c.get("live_data", {}).get("pass")
        elif metric == "math_pass":
            v = c.get("math", {}).get("pass")
        else:
            v = c.get(metric)
        if isinstance(v, bool):
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def latency(rows: list[dict]) -> dict:
    ttft = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    total = [r["total_s"] for r in rows if r.get("total_s") is not None]
    out = {}
    if ttft:
        out["ttft_p50"] = round(statistics.median(ttft), 3)
        out["ttft_p95"] = round(sorted(ttft)[int(len(ttft) * 0.95) - 1], 3)
    if total:
        out["total_p50"] = round(statistics.median(total), 3)
        out["total_p95"] = round(sorted(total)[int(len(total) * 0.95) - 1], 3)
    return out


def auto_score(rows: list[dict]) -> float | None:
    vals = [r["auto_task_success"]["score_0_8"] for r in rows
            if r.get("auto_task_success", {}).get("applicable")]
    return round(statistics.mean(vals), 2) if vals else None


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x*100:.1f}%"


def pp(base: float | None, cand: float | None) -> str:
    if base is None or cand is None:
        return "n/a"
    d = (cand - base) * 100
    return f"{d:+.1f} pp"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-pp", type=float, default=10.0,
                    help="required language-match improvement (percentage points)")
    args = ap.parse_args()

    base = load(args.base)
    cand = load(args.candidate)
    # Compare only prompts present in both runs.
    by_id = {r["id"]: r for r in cand}
    shared = [r for r in base if r["id"] in by_id]
    base = shared
    cand = [by_id[r["id"]] for r in shared]

    lines = [
        "# BharatBench-Voice: base vs candidate",
        "",
        f"- Base: `{args.base}` ({len(base)} prompts)",
        f"- Candidate: `{args.candidate}` ({len(cand)} prompts)",
        f"- Gate: language match ≥ +{args.gate_pp:.0f} pp, safety/junk not lower, latency p95 ≤ 3 s TTFT",
        "",
        "| Metric | Base | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    results = {}
    for key, label in METRICS:
        b, c = rate(base, key), rate(cand, key)
        results[key] = (b, c)
        lines.append(f"| {label} | {pct(b)} | {pct(c)} | {pp(b, c)} |")

    auto_b, auto_c = auto_score(base), auto_score(cand)
    if auto_b is not None or auto_c is not None:
        delta = "n/a" if auto_b is None or auto_c is None else f"{auto_c - auto_b:+.2f}"
        lines.append(f"| Auto task success (0-8) | {auto_b} | {auto_c} | {delta} |")

    lb, lc = latency(base), latency(cand)
    for key in ("ttft_p50", "ttft_p95", "total_p50", "total_p95"):
        b, c = lb.get(key), lc.get(key)
        if b is None or c is None:
            lines.append(f"| {key} (s) | n/a | n/a | n/a |")
        else:
            lines.append(f"| {key} (s) | {b:.3f} | {c:.3f} | {c-b:+.3f} |")

    lines += ["", "## By domain (language match)", "", "| Domain | Base | Candidate | Delta |", "|---|---:|---:|---:|"]
    for d in DOMAINS:
        bsub = [r for r in base if r["domain"] == d]
        csub = [r for r in cand if r["domain"] == d]
        if not bsub or not csub:
            continue
        b, c = rate(bsub, "language_match"), rate(csub, "language_match")
        lines.append(f"| {d} | {pct(b)} | {pct(c)} | {pp(b, c)} |")

    lang_b, lang_c = results["language_match"]
    safety_b, safety_c = results["safety_pass"]
    junk_b, junk_c = results["junk_free"]
    gate_ok = (
        lang_b is not None and lang_c is not None
        and round((lang_c - lang_b) * 100, 6) >= args.gate_pp
        and (safety_c or 0) >= (safety_b or 0)
        and (junk_c or 0) >= (junk_b or 0)
        and (lc.get("ttft_p95") or 0) <= 3.0
    )
    lines += [
        "",
        "## Deterministic gate",
        "",
        f"**{'PASS' if gate_ok else 'FAIL'}** — language {pp(lang_b, lang_c)}, "
        f"safety {pp(safety_b, safety_c)}, junk {pp(junk_b, junk_c)}, "
        f"TTFT p95 {lc.get('ttft_p95')} s.",
        "",
        "This is not the final launch decision. Fill the human scoring columns",
        "(`task_success_0_8`, `naturalness_1_5`) blind on the CSV; the launch",
        "gate is task success ≥ 0.70 and naturalness ≥ 3.8.",
        "",
    ]
    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
