#!/usr/bin/env python3
"""Vetted answer cards for high-frequency questions.

The 2B base model is fluent but unreliable on factual details: it expanded CSC
as "Computer Science and Technology", invented scheme amounts, and repeated
cooking steps. For a bounded set of common intents the system therefore returns
a curated card verbatim, like the health protocols. The model handles the long
tail.

Cards are deliberately narrow. A card matches only when its domain matches and
one of its regex patterns matches the user's text. If no card matches, the
caller falls back to the model and should show a "verify" note.

Usage
-----
  from answer_cards import CardStore
  store = CardStore()
  card = store.match("CSC center kya kaam karta hai?", "schemes")
  if card:
      print(card["answer"])
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CARDS = HERE.parent / "data" / "v4" / "cards" / "cards.jsonl"
DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    deva = len(DEVA_RE.findall(text or ""))
    latin = len(LATIN_RE.findall(text or ""))
    return "hi" if deva > latin else "hinglish"


class CardStore:
    def __init__(self, path: Path = DEFAULT_CARDS):
        self.cards: list[dict] = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    self.cards.append(json.loads(line))
        self._compiled = [
            (card, [re.compile(p, re.I) for p in card.get("patterns", [])])
            for card in self.cards
        ]

    def match(self, text: str, domain: str | None) -> dict | None:
        """Return the first matching card for the domain, or None."""
        # Small arithmetic is better computed than retrieved.
        m = re.search(r"\b(\d{1,4})\s*[x×]\s*(\d{1,4})\b", text or "")
        if m and domain in (None, "education"):
            a, b = int(m.group(1)), int(m.group(2))
            lang = detect_language(text)
            if lang == "hi":
                answer = (
                    f"{a} × {b} = {a * b}. कदम दर कदम: {a} × {b} = "
                    f"{a} × {b // 10 * 10} + {a} × {b % 10} = {a * (b // 10 * 10)} + {a * (b % 10)} = {a * b}।"
                    if b >= 10 else f"{a} × {b} = {a * b}।"
                )
            else:
                answer = (
                    f"{a} × {b} = {a * b}. Step by step: {a} × {b} = "
                    f"{a} × {b // 10 * 10} + {a} × {b % 10} = {a * (b // 10 * 10)} + {a * (b % 10)} = {a * b}. "
                    f"Yani jawab {a * b} hai."
                    if b >= 10 else f"{a} × {b} = {a * b}. Yani jawab {a * b} hai."
                )
            return {
                "id": "math-multiply",
                "domain": "education",
                "answer": answer,
                "source": "arithmetic",
                "verify": "",
                "language": lang,
            }
        # Percentages: "1000 ka 15%", "20% of 500", "25 का 40 प्रतिशत".
        m = (
            re.search(r"(\d{1,6})\s*(?:ka|का)\s*(\d{1,3})\s*(?:%|percent|pratishat|प्रतिशत)", text or "", re.I)
            or re.search(r"(\d{1,3})\s*(?:%|percent|pratishat|प्रतिशत)\s*(?:of|ka|का)\s*(\d{1,6})", text or "", re.I)
        )
        if m:
            if re.search(r"(?:%|percent|pratishat|प्रतिशत)", m.group(0), re.I):
                if "ka" in m.group(0).lower() or "का" in m.group(0):
                    whole, pct = int(m.group(1)), int(m.group(2))
                else:
                    pct, whole = int(m.group(1)), int(m.group(2))
                value = whole * pct / 100
                value_s = f"{value:g}"
                lang = detect_language(text)
                if lang == "hi":
                    answer = f"{whole} का {pct}% = {whole} × {pct}/100 = {value_s}।"
                else:
                    answer = f"{whole} ka {pct}% = {whole} × {pct}/100 = {value_s}."
                return {
                    "id": "math-percent",
                    "domain": "education",
                    "answer": answer,
                    "source": "arithmetic",
                    "verify": "",
                    "language": lang,
                }
        for card, patterns in self._compiled:
            if domain and card.get("domain") != domain:
                continue
            for pattern in patterns:
                if pattern.search(text or ""):
                    language = detect_language(text)
                    return {
                        "id": card["id"],
                        "domain": card["domain"],
                        "answer": card.get(language) or card.get("hinglish", ""),
                        "source": card.get("source", ""),
                        "verify": card.get("verify", ""),
                        "language": language,
                    }
        return None

    def __len__(self) -> int:
        return len(self.cards)


if __name__ == "__main__":
    import sys

    store = CardStore()
    print(f"cards: {len(store)}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        domain = sys.argv[1] if len(sys.argv) > 1 else None
        card = store.match(line, domain)
        print(f"{card['id'] if card else '-':28s}\t{line[:90]}")
