#!/usr/bin/env python3
"""Build v3.2: 3K curated rows."""
import json, random
from datasets import load_dataset, Dataset
random.seed(42)

def bad(msgs):
    txt=" ".join(m.get("content","") for m in msgs)
    return any(c in txt for c in ["𣷭","𫟦","𪤗","珊","�"]) or len(txt)<20 or len(txt)>2000

rows, seen=[], set()
def add(msgs):
    k=hash(json.dumps(msgs,ensure_ascii=False)[:300])
    if k in seen or bad(msgs): return
    seen.add(k); rows.append({"messages":msgs})

# 1) Evol Hindi 1.5K
print("evol-hindi...")
d=load_dataset("FreedomIntelligence/evol-instruct-hindi", split="train")
for r in d:
    if len(rows)>=1500: break
    conv=r.get("conversations",[])
    msgs=[]
    for m in conv:
        role="user" if m.get("from")=="human" else "assistant"
        msgs.append({"role":role,"content":str(m.get("value",""))})
    if len(msgs)>=2: add(msgs[:4])
print(f"  {len(rows)}")

# 2) Hinglish 1K
print("hinglish...")
d=load_dataset("Sujalvc/hinglish-instruct-dataset", split="train")
for r in d:
    if len(rows)>=2500: break
    user=r.get("instruction","")+("\n"+r.get("input","") if r.get("input") else "")
    if user.strip():
        add([{"role":"user","content":user},{"role":"assistant","content":str(r.get("output",""))}])
print(f"  {len(rows)}")

# 3) Identity 0.5K
ids=[("तुम कौन हो?","मैं Bharat-Tiny-LLM हूँ — हिंदी/Hinglish सहायक।"),("Who are you?","I am Bharat-Tiny-LLM by EvolucentAI."),("आपका नाम?","मेरा नाम Bharat-Tiny-LLM है।")]
for q,a in ids*167:
    add([{"role":"user","content":q},{"role":"assistant","content":a}])
    if len(rows)>=3000: break

random.shuffle(rows); rows=rows[:3000]
print(f"total {len(rows)}")
Dataset.from_list(rows).push_to_hub("eulogik/bharat-v3-chat-sft-v2", token="***REMOVED_HF_TOKEN***")
print("pushed")
