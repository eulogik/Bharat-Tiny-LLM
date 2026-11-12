#!/usr/bin/env python3
"""Cheap text-degeneracy checks for generated answers.

A small model can fall into a phrase loop ("chhote chhote chhote ...") that
passes character-level junk checks and still counts as a fluent answer. This
module measures repeated 5-grams and provides a threshold used by the runner
to retry once with a repetition penalty.

Usage
-----
  from text_quality import loop_ratio, is_degenerate
  if is_degenerate(answer):
      ... retry with repeat_penalty=1.15 ...
"""

from __future__ import annotations

import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F]+")
DEFAULT_THRESHOLD = 0.50
MIN_TOKENS = 30


def loop_ratio(text: str, n: int = 5) -> float:
    """Return the share of the text covered by its most repeated n-gram."""
    toks = WORD_RE.findall((text or "").lower())
    if len(toks) < max(MIN_TOKENS, n + 1):
        return 0.0
    counts = Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if not counts:
        return 0.0
    return counts.most_common(1)[0][1] / (len(toks) - n + 1)


def repeat_stats(text: str, n: int = 4) -> tuple[int, float]:
    """Return (max_count, ratio) for the most repeated n-gram."""
    toks = WORD_RE.findall((text or "").lower())
    if len(toks) < max(MIN_TOKENS, n + 1):
        return 0, 0.0
    counts = Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if not counts:
        return 0, 0.0
    count = counts.most_common(1)[0][1]
    return count, count / max(1, len(toks) - n + 1)


def is_degenerate(text: str, threshold: float = DEFAULT_THRESHOLD, n: int = 5) -> bool:
    """True if the text loops at the phrase level.

    Two signals are used:
    - a dominant 5-gram (classic loop), or
    - the same 4-gram repeated 5+ times in a long answer (a list that lost its
      content and keeps repeating the same step, which the 5-gram ratio misses).
    """
    toks = WORD_RE.findall((text or "").lower())
    if len(toks) < MIN_TOKENS:
        return False
    if loop_ratio(text, n=n) >= threshold:
        return True
    if len(toks) >= 60:
        count, ratio = repeat_stats(text, n=4)
        return count >= 5 and ratio >= 0.06
    return False


if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        line = line.strip()
        if line:
            print(f"{loop_ratio(line):.3f}\t{line[:100]}")
