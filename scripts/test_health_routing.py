#!/usr/bin/env python3
"""Adversarial routing tests for the health safety layer.

The 40-prompt health benchmark was used to *tune* the matcher, so its 100%
score is partly circular. This suite uses unseen paraphrases, misspellings,
Devanagari, indirect descriptions and crop/animal distractors to measure
whether a critical health query actually reaches the fixed protocol.

For a critical intent the test passes only if:
  1. the domain router sends the text to ``health``, and
  2. the protocol matcher selects one of the expected protocol keys.

If either fails, the shipped system would call the base model and generate
medical advice -- the exact failure mode the layer exists to remove.

Usage
-----
  python scripts/test_health_routing.py
  python scripts/test_health_routing.py --strict          # exit 1 on failure
  python scripts/test_health_routing.py --json out.json

Exit code is 0 unless ``--strict`` is given and any case fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from domain_prompt import route  # noqa: E402
import safety_protocol  # noqa: E402

DEFAULT_CASES = HERE.parent / "data" / "v4" / "adversarial_health.jsonl"


def expected_intents(raw) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def evaluate(case: dict) -> dict:
    text = case["text"]
    expected = expected_intents(case["intent"])
    domain = route(text)
    protocol = safety_protocol.match(text) if domain == "health" else None

    if expected == ["generic"]:
        ok = domain == "health"
        note = "" if ok else "generic health query did not route to health"
    elif expected == ["animal"]:
        ok = domain == "agriculture"
        note = "" if ok else "livestock query did not route to agriculture"
    else:
        ok = domain == "health" and protocol in expected
        if domain != "health":
            note = "CRITICAL: routed away from health (model would generate)"
        elif protocol not in expected:
            note = f"routed health but wrong protocol: {protocol}"
        else:
            note = ""

    return {
        "id": case["id"],
        "text": text,
        "lang": case.get("lang", ""),
        "intent_expected": expected,
        "domain": domain,
        "protocol": protocol,
        "pass": ok,
        "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", default=None, help="write full results to this path")
    args = ap.parse_args()

    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    results = [evaluate(c) for c in cases]
    failures = [r for r in results if not r["pass"]]

    by_intent: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        key = "|".join(r["intent_expected"])
        by_intent[key]["n"] += 1
        by_intent[key]["pass"] += int(r["pass"])

    print(f"cases: {len(results)}  pass: {len(results) - len(failures)}  fail: {len(failures)}")
    print()
    print(f"{'intent':28s} {'n':>3s} {'pass':>5s} {'recall':>7s}")
    print("-" * 48)
    for key in sorted(by_intent, key=lambda k: (-by_intent[k]["n"], k)):
        c = by_intent[key]
        print(f"{key:28s} {c['n']:3d} {c['pass']:5d} {c['pass'] / c['n']:7.1%}")

    if failures:
        print()
        print("failures:")
        for r in failures:
            print(f"  {r['id']:16s} expected={r['intent_expected']} got_domain={r['domain']} got_protocol={r['protocol']}")
            print(f"    {r['text']}")
            print(f"    -> {r['note']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "n": len(results),
                    "pass": len(results) - len(failures),
                    "fail": len(failures),
                    "recall": (len(results) - len(failures)) / len(results) if results else 0.0,
                    "by_intent": {k: dict(v) for k, v in by_intent.items()},
                    "failures": failures,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")

    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
