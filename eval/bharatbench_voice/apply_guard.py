#!/usr/bin/env python3
"""Apply the crop safety guard and re-score a BharatBench-Voice run offline.

The guard is a deterministic post-processor, so applying it to an existing run
is equivalent to running it live. This also re-scores with the latest safety
regexes (the agriculture "dhoyen" fix), without re-running the model.

Usage
-----
  python eval/bharatbench_voice/apply_guard.py \
    --run eval/bharatbench_voice/runs/system_full.jsonl \
    --out eval/bharatbench_voice/runs/system_full_guarded.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))

from run_bench import (  # noqa: E402
    auto_task_success,
    check_junk,
    check_language,
    check_live_data,
    check_math,
    check_safety,
    summarize,
    write_sheet,
    write_summary_md,
)
import safety_protocol  # noqa: E402
from domain_prompt import route  # noqa: E402
from text_quality import loop_ratio  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-guard", action="store_true",
                    help="re-score only, do not apply the crop guard")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.run).read_text().splitlines() if line.strip()]
    out_rows = []
    guarded = 0
    refreshed_health = 0
    for r in rows:
        r = dict(r)
        # The health path is deterministic and never calls the model, so it can
        # be recomputed exactly. Non-health rows used the same base prompt for
        # every domain, so their stored response is still valid.
        routed = route(r.get("text", ""))
        r["routed_domain"] = routed
        if not args.no_guard and routed == "health" and r.get("domain") != "animal":
            text, meta = safety_protocol.health_answer(r.get("text", ""))
            if text.strip() != r.get("response", "").strip():
                refreshed_health += 1
            r["response"] = text
            r["safety_protocol"] = meta.get("protocol")
            r["health_layer"] = "protocol" if meta.get("protocol") else "generic"
            r["crop_guard"] = False
        elif not args.no_guard and r.get("domain") == "agriculture":
            lang = safety_protocol.detect_language(r.get("text", ""))
            r["response"], replaced = safety_protocol.guard_crop(r.get("response", ""), lang, r.get("text", ""))
            r["crop_guard"] = replaced
            guarded += int(replaced)
        else:
            r["crop_guard"] = False
        r["loop_ratio"] = round(loop_ratio(r.get("response", "")), 3)
        r["loop_retry"] = bool(r.get("loop_retry", False))
        lang_ok, lang_note = check_language(r, r.get("response", ""))
        junk_ok, junk_hits = check_junk(r.get("response", ""))
        checks = {
            "language_match": lang_ok,
            "language_note": lang_note,
            "junk_free": junk_ok,
            "junk_hits": junk_hits,
            "safety": check_safety(r, r.get("response", "")),
            "live_data": check_live_data(r, r.get("response", "")),
            "math": check_math(r, r.get("response", "")),
        }
        r["checks"] = checks
        r["auto_task_success"] = auto_task_success(r, r.get("response", ""), checks)
        out_rows.append(r)

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    write_sheet(out_rows, out.with_suffix(".csv"))
    summary = summarize(out_rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(summary, out.with_suffix(".summary.md"))
    print(json.dumps({"guarded_answers": guarded, "refreshed_health_rows": refreshed_health, **summary}, indent=2))
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
