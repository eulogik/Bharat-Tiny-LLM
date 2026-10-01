#!/usr/bin/env python3
"""Build v3.3: 500 hand-picked quality rows."""
import json, os, random
from datasets import load_dataset, Dataset
random.seed(42)

def quality_utt(s):
    if not s or len(s)<15 or len(s)>600: return False
    if any(c in s for c in ["𣷭","𫟦","𪤗","珊","�","✅","checklist"]): return False
    # Must be Hindi/Hinglish-ish
    hindi=sum(1 for c in s if '\u0900' <=c<='\u097f')
    latin=sum(1 for c in s.lower() if 'a'<=c<='z')
    if hindi>5 or (hindi>0 and latin>5): return True
    return False

def bad_pair(q,a):
    if "शिव" in a and "नाम" in q: return True
    if len(a.split())<4: return True
    return False

rows=[]
# High-quality Hinglish from evol-hindi (filtered)
print("evol-hindi filtered...")
d=load_dataset("FreedomIntelligence/evol-instruct-hindi", split="train")
for r in d:
    if len(rows)>=350: break
    conv=r.get("conversations",[])
    msgs=[]
    for m in conv:
        role="user" if m.get("from")=="human" else "assistant"
        msgs.append({"role":role,"content":str(m.get("value","")).strip()})
    if len(msgs)>=2 and quality_utt(msgs[0]["content"]) and quality_utt(msgs[1]["content"]) and not bad_pair(msgs[0]["content"], msgs[1]["content"]):
        # Keep short, clean pairs
        msgs[1]["content"]=msgs[1]["content"][:600]
        rows.append({"messages":msgs[:2]})

# Add 150 Hinglish instruct (best)
print(f"  {len(rows)} so far")
d=load_dataset("Sujalvc/hinglish-instruct-dataset", split="train")
for r in d:
    if len(rows)>=480: break
    q=r.get("instruction","").strip()+("\n"+r.get("input","").strip() if r.get("input","").strip() else "")
    a=r.get("output","").strip()[:600]
    if quality_utt(q) and quality_utt(a) and not bad_pair(q,a):
        rows.append({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})

# 20 identity
ids=[("तुम कौन हो?","मैं Bharat-Tiny-LLM हूँ — EvolucentAI का हिंदी/Hinglish सहायक।"),("आपका नाम?","मेरा नाम Bharat-Tiny-LLM है।"),("What is your name?","I am Bharat-Tiny-LLM by EvolucentAI.")]
for q,a in (ids*7)[:20]:
    rows.append({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})

random.shuffle(rows); rows=rows[:500]
print(f"total {len(rows)}")
Dataset.from_list(rows).push_to_hub("eulogik/bharat-v3-chat-sft-v3", token=os.environ.get("HF_TOKEN", ""))
print("pushed v3")
