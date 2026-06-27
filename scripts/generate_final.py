#!/usr/bin/env python3 -u
"""Fast Hinglish data generator with rate-limit handling"""
import json, os, re, time, random, sys

API_KEY = open(".env").read().split("OPENROUTER_API_KEY=")[1].split("\n")[0].strip()
import requests
session = requests.Session()
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "openrouter/free"

TOPICS = [
    "chai coffee snacks cutting chai", "phone battery wifi data recharge",
    "job interview salary resume promotion", "train flight hotel travel luggage",
    "fever cold headache doctor medicine", "Diwali Holi Eid festival wedding",
    "Amazon Flipkart shopping return refund", "exam study college admission assignment",
    "UPI bank loan credit card saving", "cricket movie Netflix YouTube music",
    "biryani pizza dosa recipe cooking", "friend relationship marriage break up",
    "gym yoga workout meditation diet", "work from home meeting deadline boss",
    "birthday anniversary party gift celebration", "rain winter summer weather AC",
    "car bike driving traffic parking", "Instagram photo reel viral video",
    "morning routine sleep alarm tired", "parents family siblings home dinner",
]

MARKERS = ["yaar","bhai","hai","hoon","kya","kaise","theek","nahi","haan",
           "chai","bahut","accha","mera","karo","arre","waah","chalo","dekho","bolo","aaja","lekin"]

def gen_batch(topic, n=4):
    prompt = (
        f'Generate {n} Hinglish conversations about: {topic}.\n'
        'USER asks in Hinglish (Hindi+English mix), ASSISTANT replies helpfully in 2-3 sentences.\n'
        'Use words: yaar, bhai, haan, nahi, kya, kaise, theek, arre, accha\n'
        'Output ONLY JSON: [{"user":"...","assistant":"..."}]'
    )
    for attempt in range(8):
        try:
            resp = session.post("https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json={
                    "model": MODEL, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500, "temperature": 0.8
                }, timeout=30)
            if resp.status_code == 429:
                time.sleep(min(2**attempt, 20) + random.random()*2)
                continue
            if resp.status_code != 200:
                return []
            content = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\[.*\]', content, re.DOTALL)
            data = json.loads(m.group()) if m else json.loads(content)
            if not isinstance(data, list): return []
            return [d for d in data if isinstance(d, dict) and "user" in d and "assistant" in d
                    and len(d["user"]) > 8 and len(d["assistant"]) > 10]
        except:
            time.sleep(2**attempt)
    return []

def good(c):
    text = (c["user"] + " " + c["assistant"]).lower()
    return sum(1 for m in MARKERS if m in text) >= 1 and len(c["user"]) > 10 and len(c["assistant"]) > 20

# Count existing
count = 0
if os.path.exists("data/synthetic/hinglish_conversations.jsonl"):
    count = sum(1 for _ in open("data/synthetic/hinglish_conversations.jsonl"))
print(f"Initial: {count}", flush=True)

new = 0
start = time.time()
target = 5000

while new < target:
    topic = random.choice(TOPICS)
    t0 = time.time()
    result = gen_batch(topic, 4)
    quality = [c for c in result if good(c)]
    
    if quality:
        with open("data/synthetic/hinglish_conversations.jsonl", "a") as f:
            for c in quality:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        new += len(quality)
    
    elapsed = time.time() - start
    rate = new/elapsed*3600 if elapsed > 0 else 0
    print(f"gen={len(result)} qual={len(quality)} total={new}/{target} api={time.time()-t0:.0f}s rate={rate:.0f}/hr", flush=True)
    time.sleep(0.5 + random.random())

print(f"\nDone: {new} new in {time.time()-start:.0f}s", flush=True)
