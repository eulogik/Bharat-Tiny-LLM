#!/usr/bin/env python3
"""Build Bharat-Tiny-LLM v3.4 gold-standard mixed SFT dataset.

Sources (all quality-filtered, deduped):
  1. GSM8K-hindi train (math, #### answers only + digit-ending relaxed)
  2. Original v3 SFT non-Yojana subset (Hinglish math/code/instruct)
     - DROP all Yojana rows (44% truncated garbage + scrape junk)
  3. Chat SFT v1 (20K) + v2 (2.5K) + v3 (370) with quality filters
  4. Identity rows (model self-introduction)

Output: parquet with `messages` column, pushed to eulogik/bharat-v3-mixed-sft-v34
"""
import gzip, json, random, re, sys
from collections import Counter
from pathlib import Path

import pandas as pd

random.seed(42)

# ── quality gates ──────────────────────────────────────────────
JUNK_CHARS = ["\uFFFD", "✅", "Winvalid"]  # replacement char, checkmark; NO empty string
FOREIGN_RE = re.compile(r"[\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF\u0370-\u03FF\u0400-\u04FF]{4,}")
SCRAPE_MARKERS = [
    "are you sure you want to sign",
    "seligibility", "sexclusions", "application processdocuments",
    "<html", "</html>", "<script", "<style",
    "skip to content", "cookie policy", "© copyright",
    "all rights reserved", "frequently asked questionssources",
    "images description", "image description",
]
META_MARKERS = ["as an ai", "i am an ai", "as a language model", "lorem ipsum"]
DEV_RE = re.compile(r"[\u0900-\u097F]")


def content_str(c):
    if isinstance(c, str):
        return c
    if c is None:
        return ""
    return str(c)


def has_junk_char(s):
    return any(ch in s for ch in JUNK_CHARS)


def has_scrape(s):
    sl = s.lower()
    return any(m in sl for m in SCRAPE_MARKERS)


def has_foreign_run(s):
    return bool(FOREIGN_RE.search(s))


def has_meta(s):
    sl = s.lower()
    return any(m in sl for m in META_MARKERS)


def dev_count(s):
    return len(DEV_RE.findall(s))


def latin_count(s):
    return len(re.findall(r"[a-zA-Z]", s))


def is_hindi_hinglish(s):
    """At least some Devanagari OR romanized-Hindi signal."""
    d, l = dev_count(s), latin_count(s)
    if d >= 5:
        return True
    # romanized hindi heuristic: devanagari absent but latin present with common hindi words
    if d == 0 and l >= 20:
        sl = s.lower()
        hits = sum(w in sl for w in ["kya", "hai", "karo", "kaise", "nahi", "aap", "tum", "mera", "tera", "kar", "ho", "se", "ke", "ko", "mein", "main", "bhi", "toh", "aur", "yaar", "bhai"])
        return hits >= 3
    return d > 0


def get_msgs(obj):
    m = obj
    # unwrap nested {"messages": ...} repeatedly (sft_data has double nesting)
    for _ in range(5):
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                break
        if hasattr(m, "tolist"):
            try:
                m = m.tolist()
            except Exception:
                break
        if isinstance(m, dict) and "messages" in m:
            m = m["messages"]
        else:
            break
    return m


def validate_conversation(msgs, min_turns=2, max_chars=6000):
    """Return (ok, reason)."""
    if not isinstance(msgs, list) or len(msgs) < min_turns:
        return False, "too_few_turns"
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "assistant" not in roles or "user" not in roles:
        return False, "missing_role"
    full = "\n".join(content_str(m.get("content", "")) for m in msgs if isinstance(m, dict))
    if len(full) < 30:
        return False, "too_short"
    if len(full) > max_chars:
        return False, "too_long"
    if has_junk_char(full):
        return False, "junk_char"
    if has_scrape(full):
        return False, "scrape"
    if has_foreign_run(full):
        return False, "foreign_script"
    if has_meta(full):
        return False, "meta"
    # last message should be assistant (for SFT completion loss)
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return False, "not_assistant_ending"
    a = content_str(last.get("content", ""))
    if len(a.strip()) < 15:
        return False, "short_assistant"
    # assistant shouldn't be a pure echo of user
    u = ""
    for m in reversed(msgs[:-1]):
        if isinstance(m, dict) and m.get("role") == "user":
            u = content_str(m.get("content", ""))
            break
    if u and a.strip() == u.strip():
        return False, "echo"
    if u and len(a) < 20 and u[:40] in a:
        return False, "near_echo"
    # language: user or assistant should carry hindi/hinglish signal for chat rows
    # (math rows may be English-transliterated; accept if either side has signal)
    if not (is_hindi_hinglish(u) or is_hindi_hinglish(a) or dev_count(full) > 10):
        return False, "no_hindi_signal"
    # repetition: same line 3+ times
    lines = [ln.strip() for ln in a.split("\n") if len(ln.strip()) > 12]
    if lines:
        c = Counter(lines)
        if max(c.values()) >= 3:
            return False, "line_repeat"
    return True, "ok"


def make_gsm_row(q, ans):
    """GSM8K-hindi -> messages. Keep CoT answer; require #### or digit ending."""
    ans = str(ans).strip()
    q = str(q).strip()
    if len(q) < 20 or len(ans) < 20:
        return None, "too_short"
    if "####" not in ans and not re.search(r"\d\s*$", ans):
        return None, "no_answer"
    if has_junk_char(q) or has_junk_char(ans):
        return None, "junk"
    if has_foreign_run(q) or has_foreign_run(ans):
        return None, "foreign"
    if dev_count(q) < 10:
        return None, "not_hindi"
    # normalize: ensure assistant ends with the final answer line
    msgs = [
        {"role": "system", "content": "तुम एक गणित सहायक हैं। हल हिंदी में चरण-दर-चरण दो और अंत में उत्तर दो।"},
        {"role": "user", "content": q},
        {"role": "assistant", "content": ans},
    ]
    return {"messages": msgs}, "ok"


def dedup_key(msgs):
    u = ""
    for m in msgs:
        if m.get("role") == "user":
            u = content_str(m.get("content", ""))[:200]
            break
    a = content_str(msgs[-1].get("content", ""))[:200]
    return (u.lower().strip(), a.lower().strip())


def main():
    out_rows = []
    seen = set()
    stats = Counter()

    def add(row, source):
        nonlocal out_rows
        msgs = row["messages"]
        ok, reason = validate_conversation(msgs)
        if not ok:
            stats[f"{source}:drop_{reason}"] += 1
            return False
        k = dedup_key(msgs)
        if k in seen:
            stats[f"{source}:dup"] += 1
            return False
        seen.add(k)
        out_rows.append({"messages": msgs, "source": source})
        stats[f"{source}:keep"] += 1
        return True

    # ── 1. GSM8K-hindi ─────────────────────────────────────────
    print("=== GSM8K-hindi ===", flush=True)
    gsm = pd.read_parquet("/tmp/gsm_audit2/data/train_main-00000-of-00001.parquet")
    for _, r in gsm.iterrows():
        row, reason = make_gsm_row(r["question"], r["answer"])
        if row:
            add(row, "gsm8k_hi")
        else:
            stats[f"gsm8k_hi:drop_{reason}"] += 1
    print(f"  after: {stats['gsm8k_hi:keep']}", flush=True)

    # ── 2. Original v3 SFT non-Yojana ──────────────────────────
    print("=== v3 SFT (non-Yojana only) ===", flush=True)
    with gzip.open("/tmp/sft_art/sft_data.jsonl.gz", "rt") as f:
        for line in f:
            raw = json.loads(line)
            msgs = get_msgs(raw)
            if not isinstance(msgs, list):
                stats["v3sft:drop_struct"] += 1
                continue
            # identify Yojana by system prompt
            sys_text = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "system":
                    sys_text = content_str(m.get("content", ""))
                    break
            if "Yojana Sahayak" in sys_text or "welfare schemes" in sys_text:
                stats["v3sft:drop_yojana"] += 1
                continue
            add({"messages": msgs}, "v3sft")
    print(f"  after: {stats['v3sft:keep']}", flush=True)

    # ── 3. Chat SFT v3 (370, best curated) ─────────────────────
    print("=== chat v3 ===", flush=True)
    df3 = pd.read_parquet("/tmp/audit_v3/data/train-00000-of-00001.parquet")
    for _, r in df3.iterrows():
        msgs = get_msgs(r["messages"])
        if isinstance(msgs, list):
            add({"messages": msgs}, "chat_v3")
    print(f"  after: {stats['chat_v3:keep']}", flush=True)

    # ── 4. Chat SFT v2 (2.5K) ─────────────────────────────────
    print("=== chat v2 ===", flush=True)
    df2 = pd.read_parquet("/tmp/audit_v2/data/train-00000-of-00001.parquet")
    for _, r in df2.iterrows():
        msgs = get_msgs(r["messages"])
        if isinstance(msgs, list):
            add({"messages": msgs}, "chat_v2")
    print(f"  after: {stats['chat_v2:keep']}", flush=True)

    # ── 5. Chat SFT v1 (20K) ──────────────────────────────────
    print("=== chat v1 ===", flush=True)
    df1 = pd.read_parquet("/tmp/audit_v1c/data/train-00000-of-00001.parquet")
    for _, r in df1.iterrows():
        msgs = get_msgs(r["messages"])
        if isinstance(msgs, list):
            add({"messages": msgs}, "chat_v1")
    print(f"  after: {stats['chat_v1:keep']}", flush=True)

    # ── 6. Identity rows ───────────────────────────────────────
    print("=== identity ===", flush=True)
    ids = [
        ("तुम कौन हो?", "मैं Bharat-Tiny-LLM हूँ — EvolucentAI का हिंदी/Hinglish सहायक।"),
        ("आपका नाम क्या है?", "मेरा नाम Bharat-Tiny-LLM है।"),
        ("What is your name?", "I am Bharat-Tiny-LLM by EvolucentAI."),
        ("तुम्हारा नाम क्या है?", "मेरा नाम Bharat-Tiny-LLM है और मैं हिंदी और Hinglish में बात कर सकता हूँ।"),
        ("Who are you?", "I am Bharat-Tiny-LLM, a bilingual Hindi/Hinglish assistant by EvolucentAI."),
        ("क्या तुम हिंदी बोल सकते हो?", "हाँ, मैं हिंदी और Hinglish दोनों में बात कर सकता हूँ।"),
    ]
    for q, a in ids * 5:
        add({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}, "identity")
    print(f"  after: {stats['identity:keep']}", flush=True)

    # ── report ─────────────────────────────────────────────────
    print("\n=== STATS ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\nTOTAL KEPT: {len(out_rows)}")

    # source mix
    src_c = Counter(r["source"] for r in out_rows)
    print("mix:", dict(src_c))

    # ── write ──────────────────────────────────────────────────
    # strip source column for training (keep only messages) but save stats
    train_rows = [{"messages": r["messages"]} for r in out_rows]
    df_out = pd.DataFrame(train_rows)
    # messages column: store as list of dicts (parquet handles nested)
    out_path = Path("/tmp/v34_mixed_sft.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({out_path.stat().st_size:,} bytes, {len(df_out)} rows)")

    # also save source mix + stats
    meta = {"total": len(out_rows), "sources": dict(src_c), "stats": dict(stats)}
    Path("/tmp/v34_mixed_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote /tmp/v34_mixed_meta.json")


if __name__ == "__main__":
    main()
