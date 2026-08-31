#!/usr/bin/env python3
"""Build conversational SFT dataset for Bharat-Tiny-LLM v3 chat training.

Sources:
  - sarvamai/samvaad-hi-v1 (multi-turn Hindi, 10K rows)
  - ankitdhiman/hinglish-conversations (Hinglish, 5K rows)
  - train_gold_v3.jsonl (our corpus, 5K filtered single-turn)

Output: HF dataset eulogik/bharat-v3-chat-sft (messages format)
"""
import json, random, re, sys
from pathlib import Path
from datasets import load_dataset, Dataset

random.seed(42)

# ── helpers ──────────────────────────────────────────────────────────
def is_hindi(text):
    """Returns True if text is primarily Devanagari script."""
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097f')
    total = len(text.replace(' ', ''))
    return total > 0 and deva / total > 0.3

def is_hinglish(text):
    """Returns True if text mixes Hindi and Latin scripts."""
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097f')
    latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total = len(text.replace(' ', ''))
    if total == 0:
        return False
    return deva / total > 0.15 and latin / total > 0.15

def quality_filter(messages):
    """Filter out low-quality conversations."""
    if not messages or len(messages) < 2:
        return False
    for m in messages:
        c = m.get("content", "")
        if not c or len(c.strip()) < 5:
            return False
        # skip boilerplate
        lower = c.lower()
        if any(x in lower for x in ["✅", "checklist", "10-point plan", "ready to roll"]):
            return False
        # skip leaked system prompts
        if "new telugu instructions" in lower or "new hindi instructions" in lower:
            return False
    return True

def truncate_conversation(messages, max_turns=6, max_chars=800):
    """Keep conversations to max_turns turns and max_chars per message."""
    out = []
    for m in messages[:max_turns * 2]:
        c = m.get("content", "")[:max_chars]
        if c.strip():
            out.append({"role": m.get("role", "user"), "content": c.strip()})
    return out if len(out) >= 2 else []

# ── samvaad-hi-v1 ────────────────────────────────────────────────────
print("Loading samvaad-hi-v1...", flush=True)
samvaad = load_dataset("sarvamai/samvaad-hi-v1", split="train", trust_remote_code=True)
print(f"  raw rows: {len(samvaad)}")

samvaad_rows = []
for i, row in enumerate(samvaad):
    msgs = row.get("messages", row.get("conversations", []))
    # normalize format
    norm = []
    for m in msgs:
        role = m.get("role", m.get("from", "user"))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        else:
            continue
        content = m.get("content", m.get("value", ""))
        norm.append({"role": role, "content": content})
    norm = truncate_conversation(norm)
    if quality_filter(norm):
        samvaad_rows.append({"messages": norm})

print(f"  kept: {len(samvaad_rows)}")
del samvaad

# ── hinglish-conversations ──────────────────────────────────────────
print("Loading hinglish-conversations...", flush=True)
hinglish = load_dataset("ankitdhiman/hinglish-conversations", split="train", trust_remote_code=True)
print(f"  raw rows: {len(hinglish)}")

hinglish_rows = []
for row in hinglish:
    msgs = row.get("messages", [])
    if not msgs:
        # try turn_pairs format
        user = row.get("user_message", "")
        asst = row.get("assistant_message", "")
        if user and asst:
            msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]
    norm = []
    for m in msgs:
        role = m.get("role", m.get("from", "user"))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        else:
            continue
        content = m.get("content", m.get("value", ""))
        norm.append({"role": role, "content": content})
    norm = truncate_conversation(norm)
    if quality_filter(norm):
        hinglish_rows.append({"messages": norm})

print(f"  kept: {len(hinglish_rows)}")
del hinglish

# ── our corpus (filtered single-turn Q&A) ──────────────────────────
print("Loading train_gold_v3.jsonl...", flush=True)
corpus_path = Path("data/processed/train_gold_v3.jsonl")
our_rows = []
with open(corpus_path) as f:
    lines = f.readlines()
    # sample from diverse positions
    indices = sorted(random.sample(range(len(lines)), min(8000, len(lines))))
    for i in indices:
        try:
            row = json.loads(lines[i])
        except:
            continue
        msgs = row.get("messages", [])
        # filter: keep only clean Hindi/Hinglish single-turn pairs
        if len(msgs) < 2:
            continue
        user_msg = msgs[0].get("content", "")
        asst_msg = msgs[1].get("content", "")
        # language filter: keep Hindi/Hinglish (not pure English)
        if not (is_hindi(user_msg) or is_hinglish(user_msg)):
            continue
        # quality: response length check
        if len(asst_msg) < 20 or len(asst_msg) > 1024:
            continue
        # skip boilerplate
        if any(x in asst_msg.lower() for x in ["✅", "checklist", "10-point plan"]):
            continue
        norm = [
            {"role": "user", "content": user_msg[:800]},
            {"role": "assistant", "content": asst_msg[:800]},
        ]
        if quality_filter(norm):
            our_rows.append({"messages": norm})

print(f"  kept: {len(our_rows)}")

# ── combine & shuffle ────────────────────────────────────────────────
all_rows = samvaad_rows + hinglish_rows + our_rows
random.shuffle(all_rows)

# target ~20K rows
target = 20000
if len(all_rows) > target:
    all_rows = all_rows[:target]

print(f"\nTotal combined: {len(all_rows)}")
print(f"  samvaad: {len(samvaad_rows)}")
print(f"  hinglish: {len(hinglish_rows)}")
print(f"  our corpus: {len(our_rows)}")

# stats
total_turns = sum(len(r["messages"]) for r in all_rows)
print(f"  avg turns/conv: {total_turns/len(all_rows):.1f}")

# language distribution
deva_count = sum(1 for r in all_rows if is_hindi(r["messages"][0]["content"]))
hinglish_count = sum(1 for r in all_rows if is_hinglish(r["messages"][0]["content"]))
print(f"  language: ~{deva_count*100//len(all_rows)}% Hindi, ~{hinglish_count*100//len(all_rows)}% Hinglish")

# create dataset
ds = Dataset.from_list(all_rows)

# push to hub
repo = "eulogik/bharat-v3-chat-sft"
print(f"\nPushing to {repo}...", flush=True)
ds.push_to_hub(repo, token="***REMOVED_HF_TOKEN***")
print(f"Done! {len(ds)} rows → huggingface.co/{repo}")
