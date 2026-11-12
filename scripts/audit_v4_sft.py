#!/usr/bin/env python3
"""Audit the curated v4 SFT set before/after training.

Checks that the data actually teaches the behaviour we will claim:
  * no exact BharatBench-Voice prompt is in the training set
  * health answers mention a clinician/108 and do not prescribe
  * crop answers mention a photo/KVK and do not recommend human medicine
  * scheme answers point to an official source and avoid invented amounts/dates
  * no EvolucentAI identity, no CJK/Arabic/Thai contamination
  * language balance and duplicate rate

Usage
-----
  python scripts/audit_v4_sft.py --data data/v4/sft_curated/train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

BENCH = Path("eval/bharatbench_voice/prompts.json")

DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
DOSE_RE = re.compile(
    r"\b\d+\s?(mg|ml|tablet|goli|dawai)\b|\b(paracetamol|ibuprofen|antibiotic)\b",
    re.I,
)
HUMAN_MED_RE = re.compile(
    r"\b(paracetamol|ibuprofen|antibiotic|bandage|doctor se|doctor ko)\b", re.I
)
REFERRAL_RE = re.compile(r"\b(doctor|डॉक्टर|aspatal|hospital|अस्पताल|asha|108|104)\b", re.I)
KVK_RE = re.compile(r"\b(kvk|krishi vigyan|agriculture officer|agri officer|कृषि अधिकारी)\b", re.I)
PHOTO_RE = re.compile(r"\b(photo|फोटो|tasveer|तस्वीर)\b", re.I)
OFFICIAL_RE = re.compile(r"\b(official|portal|csc|office|site|सरकारी|पोर्टल|कार्यालय)\b", re.I)
MONEY_RE = re.compile(r"(₹|rs\.?|rupee|रुपये)\s?\d+", re.I)
EVOLU_RE = re.compile(r"evolucent", re.I)


def recommends(text: str, pattern: re.Pattern) -> bool:
    """True if the pattern appears outside a negation window."""
    negation = re.compile(r"\b(nahi|nahin|na|bina|bilkul|avoid|do not|don't|not)\b", re.I)
    for m in pattern.finditer(text):
        window = text[max(0, m.start() - 70) : m.end() + 70]
        if not negation.search(window):
            return True
    return False


def classify(text: str) -> str:
    latin = len(LATIN_RE.findall(text))
    deva = len(DEVA_RE.findall(text))
    alpha = latin + deva
    if alpha == 0:
        return "other"
    if deva / alpha >= 0.85:
        return "deva"
    if latin / alpha >= 0.85:
        return "english" if not DEVA_RE.search(text) else "mixed"
    return "mixed"


def bench_texts() -> set[str]:
    if not BENCH.exists():
        return set()
    data = json.loads(BENCH.read_text())
    return {re.sub(r"\s+", " ", p["text"].strip().lower()) for p in data["prompts"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/v4/sft_curated/train.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    bench = bench_texts()
    n = len(rows)
    langs = collections.Counter()
    sources = collections.Counter()
    problems = collections.Counter()
    health = 0
    health_ok = 0
    health_dose = 0
    crop = 0
    crop_ok = 0
    crop_human = 0
    scheme = 0
    scheme_ok = 0
    scheme_money = 0
    seen = set()

    for r in rows:
        msgs = r.get("messages", [])
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        full = user + " " + assistant
        sources[r.get("source", "?")] += 1
        langs[r.get("lang") or classify(full)] += 1

        norm = re.sub(r"\s+", " ", user.strip().lower())
        if norm in bench:
            problems["benchmark_prompt_leak"] += 1
        if EVOLU_RE.search(full):
            problems["evolucent_identity"] += 1
        if CJK_RE.search(full) or ARABIC_RE.search(full) or THAI_RE.search(full):
            problems["foreign_script"] += 1
        if len(assistant) < 20:
            problems["short_answer"] += 1
        if assistant.strip().lower() == user.strip().lower():
            problems["echo"] += 1

        key = (user[:100].lower(), assistant[:100].lower())
        if key in seen:
            problems["duplicate"] += 1
        seen.add(key)

        header = (r.get("source", "") + " " + user).lower()
        if r.get("source") == "health" or re.search(r"bukhar|dast|saanp|dard|khoon|fever|diarrh|snake", header):
            health += 1
            if REFERRAL_RE.search(assistant):
                health_ok += 1
            if recommends(assistant, DOSE_RE):
                health_dose += 1
        if r.get("source") == "agriculture" or re.search(r"tamatar|aloo|gehu|dhan|fasal|kheti|crop", header):
            crop += 1
            if PHOTO_RE.search(assistant) and KVK_RE.search(assistant):
                crop_ok += 1
            if recommends(assistant, HUMAN_MED_RE):
                crop_human += 1
        if r.get("source") == "schemes" or re.search(r"yojana|scheme|kisan|ayushman|ration|aadhaar|mudra", header):
            scheme += 1
            if OFFICIAL_RE.search(assistant):
                scheme_ok += 1
            if MONEY_RE.search(assistant):
                scheme_money += 1

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({a/b*100:.0f}%)" if b else "n/a"

    report = {
        "rows": n,
        "languages": dict(langs),
        "sources": dict(sources),
        "problems": dict(problems),
        "health": {
            "rows": health,
            "mentions_referral": pct(health_ok, health),
            "mentions_dose_or_drug": pct(health_dose, health),
        },
        "crop": {
            "rows": crop,
            "photo_and_kvk": pct(crop_ok, crop),
            "human_medicine": pct(crop_human, crop),
        },
        "schemes": {
            "rows": scheme,
            "official_source": pct(scheme_ok, scheme),
            "exact_rupee_amount": pct(scheme_money, scheme),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = []
    if problems.get("benchmark_prompt_leak"):
        failures.append("benchmark prompt leak")
    if problems.get("evolucent_identity"):
        failures.append("EvolucentAI identity")
    if problems.get("foreign_script"):
        failures.append("foreign-script contamination")
    if health and health_dose:
        failures.append("health dose/drug mention")
    if crop_human:
        failures.append("human medicine in crop rows")
    print("\nFAIL: " + ", ".join(failures) if failures else "\nPASS: no hard failures")


if __name__ == "__main__":
    main()
