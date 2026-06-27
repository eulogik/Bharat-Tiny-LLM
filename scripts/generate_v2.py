#!/usr/bin/env python3
"""High-volume Hinglish data generator using free OpenRouter models"""
import json, os, re, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

with open(".env") as f:
    for line in f:
        if line.startswith("OPENROUTER_API_KEY"):
            API_KEY = line.split("=", 1)[1].strip()
            break

import requests
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
]

TOPIC_POOLS = [
    # Casual/Friends
    ["chai", "coffee", "snacks", "movie plans", "cricket match", "weekend plans",
     "party invite", "friend visit", "phone call", "miss you", "gossip", "meme",
     "reel", "song recommend", "Netflix", "game night", "picnic", "walk"],
    # Food
    ["biryani recipe", "dinner ideas", "street food", "restaurant suggestion",
     "cooking tips", "diet plan", "weight loss food", "breakfast ideas",
     "lunch box", "food delivery", "cutting chai", "samosa", "pav bhaji", "dosa"],
    # Tech
    ["phone hang", "data recharge", "WiFi slow", "app crash", "screen crack",
     "battery drain", "storage full", "UPI fail", "OTP issue", "laptop slow",
     "printer jam", "TV not working", "Bluetooth issue", "password forgot"],
    # Career
    ["job search", "interview", "salary", "resume", "promotion", "office politics",
     "work from home", "side hustle", "freelancing", "startup", "meeting",
     "deadline", "boss", "team work", "layoff", "career change"],
    # Travel
    ["train ticket", "flight booking", "hotel search", "auto fare", "cab booking",
     "road trip", "hill station", "beach", "packing", "luggage", "passport",
     "visa", "rain travel", "metro card", "bus route", "local guide"],
    # Health
    ["fever", "cough", "cold", "headache", "stomach pain", "eye strain",
     "back pain", "sleep problem", "stress", "anxiety", "workout", "yoga",
     "diet", "vitamin", "doctor", "medicine", "home remedy", "weight loss"],
    # Family/Culture
    ["wedding", "Diwali", "Holi", "Eid", "Christmas", "birthday", "anniversary",
     "temple", "festival", "family dinner", "parents", "sibling", "cousin",
     "housewarming", "engagement", "baby", "gift", "pooja"],
    # Shopping
    ["Amazon sale", "Flipkart offer", "clothes", "shoes", "mobile", "laptop",
     "return", "exchange", "refund", "EMI", "card payment", "cash on delivery",
     "market", "bargain", "price", "quality check", "brand vs local"],
    # Education
    ["exam", "study", "college", "admission", "assignment", "project", "teacher",
     "class", "online course", "degree", "career after 12th", "internship"],
    # Finance
    ["UPI", "bank", "loan", "credit card", "saving", "investment", "SIP",
     "insurance", "tax", "budget", "salary", "money", "EMI", "crypto"],
]

def generate_batch(topics, n=4):
    topics_str = ", ".join(random.sample(topics, min(n, len(topics))))
    model = random.choice(MODELS)
    prompt = f"""Generate {n} realistic Hinglish conversations about: {topics_str}

Each conversation: USER asks a natural Hinglish question, ASSISTANT gives helpful advice in 2-3 sentences.

Hinglish = Hindi words + English words mixed naturally, like real Indian friends talk.
Use: yaar, bhai, haan, nahi, kya, kaise, theek, arre, accha, waah

Examples of good Hinglish:
- User: "Yaar, phone ki battery bahut fast drain ho rahi hai, kya karun?"
  Assistant: "Haan bhai, common problem hai. Pehle background apps check karo, screen brightness kam karo. Agar abhi bhi problem hai to battery health check karwa lo."

- User: "Chai peetey hain? Ek cutting chai ho jaye!"
  Assistant: "Haan haan, bilkul! Adrak wali chai banau ya masala chai? Thandi mein kadak chai ka maza hi kuch aur hai."

Output ONLY valid JSON array: [{{"user":"...", "assistant":"..."}}]"""

    try:
        resp = session.post("https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500, "temperature": 0.8, "top_p": 0.95
            }, timeout=60)
        if resp.status_code != 200:
            return []
        content = resp.json()["choices"][0]["message"]["content"]

        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            data = json.loads(content)

        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict)
                and "user" in d and "assistant" in d
                and len(d["user"]) > 10 and len(d["assistant"]) > 15]
    except:
        return []

# Main
count = sum(1 for _ in open("data/synthetic/hinglish_conversations.jsonl")) \
        if os.path.exists("data/synthetic/hinglish_conversations.jsonl") else 0
print(f"Starting with {count} conversations")

total_new = 0
attempts = 0
start_time = time.time()

# Generate in bulk using topic pools
BATCH_SIZE = 8  # topics per batch
CONVOS_PER_BATCH = 4
TARGET_NEW = 10000

with ThreadPoolExecutor(max_workers=6) as ex:
    while total_new < TARGET_NEW:
        batch_start = time.time()
        batch_futures = []
        for _ in range(20):  # 20 parallel requests
            pool = random.choice(TOPIC_POOLS)
            topics = random.sample(pool, min(BATCH_SIZE, len(pool)))
            batch_futures.append(ex.submit(generate_batch, topics, CONVOS_PER_BATCH))

        batch_results = []
        for f in as_completed(batch_futures):
            result = f.result()
            batch_results.extend(result)
            attempts += 1

        # Quality filter
        hinglish_markers = ["yaar", "bhai", "hai", "hoon", "kya", "kaise", "theek",
                           "nahi", "haan", "chai", "bahut", "accha", "mera", "karo",
                           "arre", "waah", "chalo", "dekho", "bolo", "aaja", "lekin"]
        quality = []
        for c in batch_results:
            text = (c["user"] + " " + c["assistant"]).lower()
            score = sum(1 for m in hinglish_markers if m in text)
            if score >= 1 and len(c["user"]) > 10 and len(c["assistant"]) > 20:
                quality.append(c)

        # Append to file
        with open("data/synthetic/hinglish_conversations.jsonl", "a") as f:
            for c in quality:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        total_new += len(quality)
        elapsed = time.time() - start_time
        rate = total_new / elapsed * 3600 if elapsed > 0 else 0
        print(f"  +{len(quality)} quality | {total_new}/{TARGET_NEW} | {elapsed:.0f}s | {rate:.0f}/hr")
        
        if total_new >= TARGET_NEW:
            break

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
print(f"Total API attempts: {attempts}")
print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Avg rate: {len(unique)/elapsed*3600:.0f} conversations/hour")
print(f"{'='*50}")

# Show samples
random.seed(42)
random.shuffle(unique)
for i, c in enumerate(unique[:5]):
    print(f"\n{i+1}. U: {c['user'][:120]}")
    print(f"   A: {c['assistant'][:120]}")
