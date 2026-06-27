#!/usr/bin/env python3
"""Robust Hinglish data generator with rate-limit handling"""
import json, os, re, time, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = open(".env").read().split("OPENROUTER_API_KEY=")[1].split("\n")[0].strip()
import requests
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=1))
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODEL = "openrouter/free"

TOPIC_PAIRS = [
    ("Chai, coffee, snacks, cutting chai, tapri", "food"),
    ("Phone battery, WiFi slow, data recharge, screen cracked, app crash", "tech"),
    ("Job interview, salary negotiation, resume, promotion, side hustle", "career"),
    ("Train ticket, flight booking, hotel search, luggage lost, auto fare", "travel"),
    ("Fever, cold, headache, stomach pain, home remedy, doctor visit", "health"),
    ("Diwali, Holi, Eid, wedding, birthday, anniversary, family function", "culture"),
    ("Amazon, Flipkart, shopping, return, refund, exchange, sale", "shopping"),
    ("Exam stress, study tips, college admission, online course, project", "education"),
    ("UPI, bank, loan, credit card, saving, investment, EMI", "finance"),
    ("Friend, relationship, marriage, break up, family, parents", "social"),
    ("Cricket, movie, Netflix, game, music, YouTube, Instagram", "entertainment"),
    ("Biryani, dosa, pizza, burger, pav bhaji, diet, recipe, cooking", "food"),
    ("Work from home, meeting, deadline, boss, team, office politics", "work"),
    ("Yoga, gym, workout, meditation, stress, anxiety, sleep", "wellness"),
]

def gen_batch(topic_str, n=4):
    prompt = (
        'Generate ' + str(n) + ' Hinglish conversations about: ' + topic_str + '.\n'
        'Each: USER asks in Hinglish (Hindi+English mix), ASSISTANT replies helpfully in 2-3 sentences.\n'
        'Use: yaar, bhai, haan, nahi, kya, kaise, theek, arre, accha\n\n'
        'Output ONLY a JSON array: [{"user":"...","assistant":"..."}]'
    )
    for attempt in range(5):
        try:
            resp = session.post("https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500, "temperature": 0.8
                }, timeout=30)
            
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                time.sleep(wait + random.random() * 2)
                continue
            if resp.status_code != 200:
                return []
            
            content = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\[.*\]', content, re.DOTALL)
            data = json.loads(m.group()) if m else json.loads(content)
            if not isinstance(data, list):
                return []
            return [d for d in data if isinstance(d, dict) and "user" in d and "assistant" in d
                    and len(d["user"]) > 8 and len(d["assistant"]) > 10]
        except Exception as e:
            time.sleep(2 ** attempt)
    return []

hinglish_markers = ["yaar", "bhai", "hai", "hoon", "kya", "kaise", "theek",
                   "nahi", "haan", "chai", "bahut", "accha", "mera", "karo",
                   "arre", "waah", "chalo", "dekho", "bolo", "aaja", "lekin"]

def is_quality(c):
    text = (c["user"] + " " + c["assistant"]).lower()
    score = sum(1 for m in hinglish_markers if m in text)
    return score >= 1 and len(c["user"]) > 10 and len(c["assistant"]) > 20

# Main
count = sum(1 for _ in open("data/synthetic/hinglish_conversations.jsonl")) \
        if os.path.exists("data/synthetic/hinglish_conversations.jsonl") else 0
print(f"Starting with {count} conversations")

total_new = 0
start_time = time.time()
target = 5000
good_count = 0
bad_count = 0
workers = 4

while total_new < target:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for _ in range(workers * 2):
            topic, cat = random.choice(TOPIC_PAIRS)
            futures.append(ex.submit(gen_batch, topic, 4))
        
        for f in as_completed(futures):
            result = f.result()
            quality = [c for c in result if is_quality(c)]
            bad_count += len(result) - len(quality)
            
            if quality:
                with open("data/synthetic/hinglish_conversations.jsonl", "a") as f:
                    for c in quality:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                total_new += len(quality)
                good_count += len(quality)
    
    elapsed = time.time() - start_time
    rate = total_new / elapsed * 3600 if elapsed > 0 else 0
    print(f"  +{good_count} good / {bad_count} bad | total={total_new}/{target} | {elapsed:.0f}s | {rate:.0f}/hr")
    good_count = 0
    bad_count = 0

# Final dedup
print("\n=== Deduplicating... ===")
seen = set()
unique = []
with open("data/synthetic/hinglish_conversations.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            s = json.loads(line)
            key = (s.get("user",""), s.get("assistant",""))
            if key not in seen:
                seen.add(key)
                unique.append(s)
        except: pass

with open("data/synthetic/hinglish_conversations.jsonl", "w") as f:
    for s in unique:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

elapsed = time.time() - start_time
print(f"\n{'='*50}")
print(f"FINAL: {len(unique)} unique conversations")
print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*50}")
