#!/usr/bin/env python3
"""Retrieval-only health FAQ.

The v4 SFT experiments showed that a 2B model will improvise dangerous health
advice when it is allowed to generate: burns -> "doodh ya ghee", fracture ->
"electric shock", pneumonia -> a repetition loop with no referral. For health,
the system therefore does **not** generate. It retrieves the closest vetted
answer from the curated health set and returns it verbatim, with a disclaimer.

The curated health answers were audited: 0/48 recommend a drug or dose, 44/48
mention a clinician/ASHA/108. They are not a substitute for a doctor, but they
are safe and consistent.

Usage
-----
  from health_faq import HealthFAQ
  faq = HealthFAQ()
  text, meta = faq.answer("Jalne par pehla upay kya hai?")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_SFT = Path("data/v4/sft_curated/train.jsonl")

WORD_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)

DISCLAIMER = (
    "Main doctor nahi hoon; ye general safety information hai. "
    "Apne lakshan ke liye doctor/ASHA se milein. Emergency mein 108."
)
DISCLAIMER_HI = (
    "मैं डॉक्टर नहीं हूँ; यह सामान्य सुरक्षा जानकारी है। "
    "अपने लक्षणों के लिए डॉक्टर/ASHA से मिलें। इमरजेंसी में 108।"
)

GENERIC = (
    "Main doctor nahi hoon aur lakshan dekh kar diagnosis nahi kar sakta. "
    "Kripya nazdiki doctor, ASHA worker ya PHC se milein. Agar saans phool rahi "
    "hai, seene mein dard hai, behoshi ho rahi hai, ya bahut tez bukhar hai to "
    "turant 108 par call karein."
)
GENERIC_HI = (
    "मैं डॉक्टर नहीं हूँ और लक्षण देखकर diagnosis नहीं कर सकता। "
    "कृपया नज़दीकी डॉक्टर, ASHA worker या PHC से मिलें। अगर साँस फूल रही है, "
    "सीने में दर्द है, बेहोशी हो रही है, या बहुत तेज़ बुखार है तो तुरंत 108 पर कॉल करें।"
)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_language(text: str) -> str:
    deva = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "hi" if deva > latin else "hinglish"


class HealthFAQ:
    def __init__(self, path: Path = DEFAULT_SFT):
        self.rows: list[dict] = []
        if path.exists():
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("source") != "health":
                        continue
                    msgs = obj.get("messages", [])
                    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                    assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
                    if user and assistant:
                        self.rows.append({
                            "question": user,
                            "answer": assistant,
                            "tokens": tokens(user),
                        })

    def answer(self, query: str) -> tuple[str, dict]:
        lang = detect_language(query)
        q = tokens(query)
        best = (0.0, None)
        for row in self.rows:
            score = jaccard(q, row["tokens"])
            if score > best[0]:
                best = (score, row)
        if best[1] is None or best[0] < 0.30:
            return (GENERIC_HI if lang == "hi" else GENERIC), {
                "health_faq": None,
                "score": round(best[0], 3),
            }
        row = best[1]
        disclaimer = DISCLAIMER_HI if lang == "hi" else DISCLAIMER
        return row["answer"] + "\n\n" + disclaimer, {
            "health_faq": row["question"],
            "score": round(best[0], 3),
        }


if __name__ == "__main__":
    import sys

    faq = HealthFAQ()
    print(f"Health FAQ rows: {len(faq.rows)}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        text, meta = faq.answer(line)
        print(f"\nQ: {line}\n{text}\n{meta}")
