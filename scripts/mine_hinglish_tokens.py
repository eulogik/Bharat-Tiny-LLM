#!/usr/bin/env python3
"""Mine Hinglish-first vocabulary injection candidates for Bharat-Tiny-LLM v4.

Why this exists
---------------
v2/v3 injected Devanagari subwords. That helped native-script Hindi but not the
way most users actually type: Romanized Hinglish ("bhai yeh kaise hoga?").
Qwen3.5-2B's 248K tokenizer already handles Devanagari well, but it still
splits very common Hinglish words into 2-3 tokens (hain, yeh, chahiye, bilkul).

This script:
  1. streams a messages-format JSONL corpus,
  2. classifies messages as Hinglish / English / Devanagari / mixed,
  3. mines Latin-script Hinglish word candidates that currently cost >1 token,
  4. scores candidates by measured token savings * frequency,
  5. selects a token budget (default 800 Hinglish + 200 Devanagari),
  6. validates on a held-out file by actually calling tokenizer.add_tokens(),
  7. writes a JSON report + a human-readable Markdown summary.

The validation is the important part: it measures the *real* post-injection
tokenizer, not a simulation. Run it before spending any GPU time.

Usage
-----
  venv/bin/python scripts/mine_hinglish_tokens.py \
    --corpus "/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl" \
    --valid data/processed/valid_gold.jsonl \
    --tokenizer Qwen/Qwen3.5-2B \
    --budget 1000 --hinglish-budget 800 --deva-budget 200 \
    --max-lines 200000 \
    --out-dir data/v4

Outputs
-------
  data/v4/hinglish_candidates.json   full machine-readable report
  data/v4/hinglish_tokens.json       selected tokens, ready for injection
  data/v4/hinglish_candidates.md     human-readable summary
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Hinglish / English classification
# ---------------------------------------------------------------------------

# Common Romanized Hindi function words. A message containing several of these
# is Hinglish even when the surrounding text is English.
HINGLISH_MARKERS = {
    "hai", "hain", "ha", "hain", "kya", "kyun", "kyu", "kaise", "kaisa", "kaisi",
    "nahi", "nahin", "na", "bhi", "hi", "toh", "to", "aur", "ya", "par", "lekin",
    "magar", "agar", "jab", "tab", "ab", "abhi", "phir", "fir", "kal", "aaj",
    "main", "mein", "mai", "hum", "aap", "tum", "tu", "woh", "vo", "yeh", "ye",
    "mera", "meri", "mere", "tera", "teri", "tere", "hamara", "hamari", "apna",
    "apni", "kar", "karo", "karna", "karte", "karti", "karta", "kiya", "kiye",
    "hoga", "hogi", "hote", "hoti", "hota", "raha", "rahi", "rahe", "gaya",
    "gayi", "gaye", "chahiye", "chaiye", "sakte", "sakta", "sakti", "sakta",
    "bilkul", "bahut", "bohot", "zyada", "kam", "kuch", "kucch", "sab", "sabhi",
    "ekdum", "thik", "theek", "acha", "accha", "achha", "bura", "haan", "han",
    "ji", "bhai", "yaar", "dost", "matlab", "samajh", "samjho", "bolo", "batao",
    "bata", "dekho", "dekh", "suno", "sun", "chalo", "chal", "ruko", "ruko",
    "liye", "liya", "liye", "dena", "do", "diya", "di", "lena", "lo", "liya",
    "wala", "wali", "wale", "log", "logon", "ghar", "kaam", "paisa", "paise",
    "bharat", "hindustan", "desh", "sarkar", "yojana", "kisan", "kheti", "fasal",
    "bazar", "mandi", "sehat", "dawai", "doctor", "siksha", "school", "college",
    "exam", "result", "naukri", "family", "mummy", "papa", "bhaiya", "didi",
    "beta", "beti", "maa", "papa", "dadi", "nana", "nani", "chacha", "mausi",
}

# High-frequency English words that must never be injected even if they appear
# inside Hinglish messages. Kept deliberately broad; the data-driven
# english_ratio filter below catches the rest.
ENGLISH_COMMON = {
    "the", "and", "for", "you", "your", "are", "was", "were", "have", "has",
    "had", "will", "would", "could", "should", "can", "may", "might", "must",
    "this", "that", "these", "those", "with", "from", "into", "about", "after",
    "before", "because", "while", "where", "when", "what", "which", "who",
    "whom", "whose", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "than", "too",
    "very", "just", "also", "not", "now", "out", "over", "under", "again",
    "then", "once", "here", "there", "please", "thanks", "thank", "sorry",
    "hello", "hi", "hey", "yes", "no", "okay", "ok", "good", "great", "best",
    "better", "bad", "new", "old", "big", "small", "long", "short", "high",
    "low", "first", "last", "next", "many", "much", "every", "another", "same",
    "india", "indian", "english", "hindi", "language", "model", "data", "test",
    "user", "assistant", "system", "prompt", "response", "question", "answer",
    "help", "need", "want", "like", "make", "made", "take", "give", "get",
    "got", "know", "think", "see", "look", "come", "go", "going", "use", "used",
    "work", "working", "time", "day", "year", "people", "person", "thing",
    "things", "world", "life", "home", "house", "water", "food", "health",
    "school", "college", "student", "teacher", "doctor", "money", "price",
    "market", "business", "company", "office", "email", "phone", "mobile",
    "computer", "internet", "online", "offline", "app", "apps", "chat", "bot",
    "open", "close", "start", "stop", "run", "set", "add", "remove", "update",
    "version", "file", "files", "code", "python", "linux", "windows", "mac",
    "apple", "google", "microsoft", "facebook", "whatsapp", "instagram",
    "youtube", "twitter", "linkedin", "google", "amazon", "flipkart",
    "government", "scheme", "schemes", "policy", "law", "legal", "court",
    "health", "medical", "medicine", "hospital", "doctor", "nurse", "patient",
    "crop", "farmer", "farming", "soil", "weather", "rain", "market", "price",
    "exam", "result", "question", "answer", "study", "learn", "teach",
    "casual", "actually", "basically", "seriously", "really", "pretty",
    "super", "nice", "cool", "awesome", "sure", "fine", "okay", "sorry",
    "welcome", "congrats", "congratulations", "morning", "evening", "night",
    "today", "tomorrow", "yesterday", "week", "month", "hour", "minute",
    "second", "friend", "friends", "brother", "sister", "mother", "father",
    "family", "people", "person", "someone", "anyone", "everyone", "nothing",
    "something", "anything", "everything", "maybe", "probably", "definitely",
    "absolutely", "exactly", "usually", "always", "never", "sometimes",
    "often", "again", "still", "already", "almost", "enough", "quite",
    "rather", "pretty", "little", "large", "important", "possible",
    "simple", "easy", "hard", "difficult", "happy", "sad", "angry",
    "excited", "tired", "busy", "free", "ready", "done", "finished",
    "started", "stopped", "waiting", "trying", "going", "coming", "doing",
    "saying", "telling", "asking", "giving", "taking", "making", "getting",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,23}")
DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

ENGLISH_ONLY_RATIO = 0.75
HINGLISH_LATIN_RATIO = 0.55


def classify(text: str) -> str:
    """Classify a message as hinglish / english / deva / mixed / other."""
    latin = len(LATIN_RE.findall(text))
    deva = len(DEVA_RE.findall(text))
    alpha = latin + deva
    if alpha == 0:
        return "other"
    words = [w.lower() for w in WORD_RE.findall(text)]
    markers = sum(1 for w in words if w in HINGLISH_MARKERS)
    latin_ratio = latin / alpha
    if latin_ratio >= ENGLISH_ONLY_RATIO and markers == 0:
        return "english"
    if deva / alpha >= 0.85:
        return "deva"
    if latin_ratio >= HINGLISH_LATIN_RATIO and markers >= 2:
        return "hinglish"
    return "mixed"


def iter_texts(path: Path, max_lines: int = 0) -> Iterable[str]:
    """Yield message contents from a messages-format JSONL file."""
    with path.open() as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in obj.get("messages", []):
                content = msg.get("content", "")
                if content:
                    yield content


def token_len(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def space_variant(word: str) -> str:
    return " " + word


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def mine(args) -> dict:
    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")
    valid = Path(args.valid)
    if not valid.exists():
        raise SystemExit(f"validation file not found: {valid}")

    print(f"Loading tokenizer {args.tokenizer} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Counts
    word_freq_hinglish: Counter[str] = Counter()
    word_freq_all: Counter[str] = Counter()
    word_freq_english: Counter[str] = Counter()
    deva_ngrams: dict[int, Counter[str]] = {n: Counter() for n in range(2, 7)}
    deva_words: Counter[str] = Counter()
    n_msgs = 0
    n_hinglish = 0
    n_english = 0
    n_deva = 0
    n_mixed = 0

    print(f"Streaming {corpus} (max_lines={args.max_lines or 'all'}) ...",
          file=sys.stderr, flush=True)
    for text in iter_texts(corpus, args.max_lines):
        n_msgs += 1
        kind = classify(text)
        if kind == "hinglish":
            n_hinglish += 1
        elif kind == "english":
            n_english += 1
        elif kind == "deva":
            n_deva += 1
        elif kind == "mixed":
            n_mixed += 1

        words = [w.lower() for w in WORD_RE.findall(text)]
        for w in words:
            word_freq_all[w] += 1
            if kind in ("hinglish", "mixed"):
                word_freq_hinglish[w] += 1
            elif kind == "english":
                word_freq_english[w] += 1

        # Devanagari n-grams from all text (not just deva-only messages)
        for run in re.findall(r"[\u0900-\u097F]{2,}", text):
            if len(run) <= 24:
                deva_words[run] += 1
            for n in range(2, 7):
                for i in range(len(run) - n + 1):
                    deva_ngrams[n][run[i:i + n]] += 1

        if n_msgs % 20000 == 0:
            print(f"  {n_msgs:,} messages ...", file=sys.stderr, flush=True)

    print(
        f"Messages: {n_msgs:,} | hinglish {n_hinglish:,} | english {n_english:,} "
        f"| deva {n_deva:,} | mixed {n_mixed:,}",
        file=sys.stderr,
    )

    # --- Hinglish candidates ---
    print("Scoring Hinglish word candidates ...", file=sys.stderr, flush=True)
    word_cache: dict[str, tuple[int, int, int, int]] = {}
    candidates = []
    for w, freq_h in word_freq_hinglish.items():
        if freq_h < args.min_freq:
            continue
        if len(w) < args.min_word_len:
            continue
        if w in ENGLISH_COMMON:
            continue
        freq_all = word_freq_all[w]
        freq_en = word_freq_english[w]
        if freq_all == 0:
            continue
        if freq_en / freq_all > args.max_english_ratio:
            continue
        if w not in word_cache:
            word_cache[w] = (
                token_len(tok, w),
                token_len(tok, space_variant(w)),
                token_len(tok, " " + w),
                token_len(tok, w.capitalize()),
            )
        tok_plain, tok_space, tok_space2, tok_cap = word_cache[w]
        # Leading-space tokens are the safe form for a byte-level BPE: they
        # absorb the space and cannot match inside an English word. The plain
        # form is deliberately omitted; sentence-start words are a small slice
        # and plain tokens caused English regressions in validation.
        saved = tok_space - 1
        if saved <= 0:
            continue
        # Score = total tokens saved, with a small frequency damping so a
        # single word cannot consume the whole budget.
        score = saved * math.sqrt(freq_h)
        candidates.append({
            "token": space_variant(w),
            "word": w,
            "context": "mid",
            "freq": freq_h,
            "freq_all": freq_all,
            "freq_english": freq_en,
            "current_tokens": tok_space,
            "saved_per_use": saved,
            "score": round(score, 3),
            "total_saved": saved * freq_h,
            "kind": "hinglish",
        })

    candidates.sort(key=lambda c: (-c["score"], -c["total_saved"], c["token"]))

    # Greedy selection with a per-word cap: at most 2 variants (mid + start),
    # and never more than the budget.
    selected_hinglish = []
    per_word: defaultdict[str, int] = defaultdict(int)
    seen_tokens = set()
    for c in candidates:
        if len(selected_hinglish) >= args.hinglish_budget:
            break
        if c["token"] in seen_tokens:
            continue
        if per_word[c["word"]] >= 2:
            continue
        per_word[c["word"]] += 1
        seen_tokens.add(c["token"])
        selected_hinglish.append(c)

    # --- Devanagari candidates ---
    print("Scoring Devanagari candidates ...", file=sys.stderr, flush=True)
    deva_candidates = []
    if args.deva_mode == "word":
        pool = deva_words.most_common(args.deva_pool)
        for word, freq in pool:
            if freq < args.min_freq:
                continue
            token = space_variant(word)
            ntok = token_len(tok, token)
            saved = ntok - 1
            if saved <= 0:
                continue
            deva_candidates.append({
                "token": token,
                "kind": "deva",
                "freq": freq,
                "current_tokens": ntok,
                "saved_per_use": saved,
                "score": round(saved * math.sqrt(freq), 3),
                "total_saved": saved * freq,
            })
    else:
        for n in range(2, 7):
            for ng, freq in deva_ngrams[n].most_common(args.deva_pool):
                if freq < args.min_freq:
                    continue
                token = space_variant(ng)
                ntok = token_len(tok, token)
                saved = ntok - 1
                if saved <= 0:
                    continue
                deva_candidates.append({
                    "token": token,
                    "kind": "deva",
                    "freq": freq,
                    "current_tokens": ntok,
                    "saved_per_use": saved,
                    "score": round(saved * math.sqrt(freq), 3),
                    "total_saved": saved * freq,
                })
    deva_candidates.sort(key=lambda c: (-c["score"], -c["total_saved"], c["token"]))
    selected_deva = []
    seen_deva = set()
    for c in deva_candidates:
        if len(selected_deva) >= args.deva_budget:
            break
        if c["token"] in seen_deva:
            continue
        seen_deva.add(c["token"])
        selected_deva.append(c)

    all_selected = selected_hinglish + selected_deva
    tokens = [c["token"] for c in all_selected]

    # --- Held-out validation: actually add tokens and measure ---
    print("Validating on held-out data ...", file=sys.stderr, flush=True)
    base_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    val_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    before_vocab = len(val_tok)
    added = val_tok.add_tokens(tokens, special_tokens=False)
    after_vocab = len(val_tok)

    # Materialize held-out message types
    val_msgs = list(iter_texts(valid, args.max_valid_lines))
    val_hinglish = [t for t in val_msgs if classify(t) == "hinglish"]
    val_english = [t for t in val_msgs if classify(t) == "english"]
    val_deva = [t for t in val_msgs if DEVA_RE.search(t)]

    def measure(texts: list[str], tokenizer) -> dict:
        chars = sum(len(t) for t in texts)
        toks = sum(token_len(tokenizer, t) for t in texts)
        return {
            "texts": len(texts),
            "chars": chars,
            "tokens": toks,
            "tokens_per_1000_chars": round(toks / chars * 1000, 2) if chars else None,
        }

    baseline_hinglish = measure(val_hinglish, base_tok)
    injected_hinglish = measure(val_hinglish, val_tok)
    baseline_all = measure(val_msgs, base_tok)
    injected_all = measure(val_msgs, val_tok)
    baseline_english = measure(val_english, base_tok)
    injected_english = measure(val_english, val_tok)
    baseline_deva = measure(val_deva, base_tok)
    injected_deva = measure(val_deva, val_tok)

    def savings(before: dict, after: dict) -> float | None:
        if not before["tokens"] or not after["tokens"]:
            return None
        return round((1 - after["tokens"] / before["tokens"]) * 100, 2)

    report = {
        "meta": {
            "tokenizer": args.tokenizer,
            "corpus": str(corpus),
            "valid": str(valid),
            "max_lines": args.max_lines,
            "messages_seen": n_msgs,
            "class_counts": {
                "hinglish": n_hinglish,
                "english": n_english,
                "deva": n_deva,
                "mixed": n_mixed,
            },
            "budget": args.budget,
            "hinglish_budget": args.hinglish_budget,
            "deva_budget": args.deva_budget,
            "min_freq": args.min_freq,
            "vocab_before": before_vocab,
            "vocab_after": after_vocab,
            "tokens_requested": len(tokens),
            "tokens_added": added,
        },
        "holdout": {
            "hinglish": {
                "baseline": baseline_hinglish,
                "injected": injected_hinglish,
                "savings_pct": savings(baseline_hinglish, injected_hinglish),
            },
            "all": {
                "baseline": baseline_all,
                "injected": injected_all,
                "savings_pct": savings(baseline_all, injected_all),
            },
            "english": {
                "baseline": baseline_english,
                "injected": injected_english,
                "savings_pct": savings(baseline_english, injected_english),
            },
            "devanagari": {
                "baseline": baseline_deva,
                "injected": injected_deva,
                "savings_pct": savings(baseline_deva, injected_deva),
            },
        },
        "selected": all_selected,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hinglish_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (out_dir / "hinglish_tokens.json").write_text(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "tokens": tokens,
                "hinglish_count": len(selected_hinglish),
                "deva_count": len(selected_deva),
                "expected": {
                    "hinglish_savings_pct": savings(baseline_hinglish, injected_hinglish),
                    "all_savings_pct": savings(baseline_all, injected_all),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Markdown summary
    lines = [
        "# Hinglish token injection — mining report",
        "",
        f"- Tokenizer: `{args.tokenizer}`",
        f"- Corpus: `{corpus}`",
        f"- Held-out: `{valid}`",
        f"- Messages seen: **{n_msgs:,}**",
        f"- Vocab: **{before_vocab:,} → {after_vocab:,}** (+{added})",
        f"- Budget: {args.budget} (Hinglish {args.hinglish_budget}, Devanagari {args.deva_budget})",
        "",
        "## Held-out token savings",
        "",
        "| Slice | Baseline tok/1000 chars | Injected tok/1000 chars | Savings |",
        "|---|---:|---:|---:|",
    ]
    for name, key in (
        ("Hinglish messages", "hinglish"),
        ("All messages", "all"),
        ("English-only messages", "english"),
        ("Devanagari messages (full text)", "devanagari"),
    ):
        h = report["holdout"][key]
        lines.append(
            f"| {name} | {h['baseline']['tokens_per_1000_chars']} | "
            f"{h['injected']['tokens_per_1000_chars']} | {h['savings_pct']}% |"
        )
    lines += [
        "",
        "## Top Hinglish candidates",
        "",
        "| # | Token | Current tokens | Freq | Saved/use | Score |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, c in enumerate(selected_hinglish[:40], 1):
        lines.append(
            f"| {i} | `{c['token']}` | {c['current_tokens']} | {c['freq']:,} | "
            f"{c['saved_per_use']} | {c['score']:.0f} |"
        )
    lines += ["", f"_Devanagari candidates selected: {len(selected_deva)}_", ""]
    (out_dir / "hinglish_candidates.md").write_text("\n".join(lines))

    print(json.dumps({
        "vocab_before": before_vocab,
        "vocab_after": after_vocab,
        "tokens_added": added,
        "hinglish_savings_pct": savings(baseline_hinglish, injected_hinglish),
        "all_savings_pct": savings(baseline_all, injected_all),
        "english_savings_pct": savings(baseline_english, injected_english),
        "deva_savings_pct": savings(baseline_deva, injected_deva),
        "out_dir": str(out_dir),
    }, indent=2))
    print(f"\nWrote {out_dir}/hinglish_candidates.{{json,md}}", file=sys.stderr)
    return report


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl")
    ap.add_argument("--valid", default="data/processed/valid_gold.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--hinglish-budget", type=int, default=800)
    ap.add_argument("--deva-budget", type=int, default=200)
    ap.add_argument("--max-lines", type=int, default=200_000,
                    help="0 = all lines")
    ap.add_argument("--max-valid-lines", type=int, default=0,
                    help="0 = all held-out lines")
    ap.add_argument("--min-freq", type=int, default=20)
    ap.add_argument("--min-word-len", type=int, default=3)
    ap.add_argument("--max-english-ratio", type=float, default=0.35,
                    help="drop words whose occurrences are mostly English-only")
    ap.add_argument("--deva-pool", type=int, default=4000,
                    help="top n-grams per length to score")
    ap.add_argument("--deva-mode", default="ngram", choices=["ngram", "word"],
                    help="mine Devanagari n-grams or whole words")
    ap.add_argument("--out-dir", default="data/v4")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    mine(args)
