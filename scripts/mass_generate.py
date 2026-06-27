#!/usr/bin/env python3
"""Mass-generate Hinglish training data via OpenRouter API"""
import json, os, re, time, sys
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

MODEL = "qwen/qwen3-8b"

ROUND_TOPICS = [
    # Round 1: Daily life (100 topics)
    ["chai", "samosa", "office", "boss", "salary", "EMI", "bike", "car",
     "haircut", "shirt", "shoes", "mobile game", "reel", "Netflix", "exam",
     "homework", "teacher", "principal", "picnic", "friendship", "valentine date", "first date",
     "breakup", "love marriage", "arranged marriage", "family dinner", "housewarming", "new baby", "toddler",
     "school admission", "gas booking", "ration card", "aadhar card", "bank account", "missed call", "wrong number",
     "good morning", "forwarded message", "WhatsApp status", "video", "music", "studio", "comedy",
     "open mic", "audition", "gift buying", "surprise party",
     "train journey", "flight travel", "luggage lost", "passport photo",
     "hotel checkin", "room service", "buffet breakfast", "rain dance",
     "mountain trek", "beach vacation", "camping night", "bonfire", "ghost story",
     "road rage", "fine payment", "license lost", "accident claim",
     "charity", "blood donation", "temple visit", "namaz time", "church service", "gurdwara langar",
     "yoga", "meditation", "lockdown", "mask", "vaccine", "covid",
     "work from home", "zoom meeting", "deadline", "project", "code bug", "startup idea",
     "funding", "investor", "crypto", "stock market", "mutual fund", "SIP", "insurance", "tax",
     "electricity bill", "water shortage", "internet speed", "landlord", "flatmate"][:100],

    # Round 2: Cultural & social
    [f"{a}" for a in ["Diwali shopping", "Holi colors", "Eid feast", "Christmas party",
     "Pongal celebration", "wedding reception", "engagement party", "baby shower",
     "birthday surprise", "anniversary gift", "housewarming pooja", "temple visit",
     "gurdwara langar", "church mass", "namaz time", "fasting tips", "vrat katha",
     "navratri garba", "ganesh chaturthi", "durga pooja", "Eid namaz", "holi milan",
     "diwali puja", "karwa chauth", "raksha bandhan", "bhai dooj", "chhath puja",
     "onam sadya", "makar sankranti", "kumbh mela", "tirupati visit", "gold purchase",
     "saree shopping", "jewellery", "mehendi", "sangeet", "haldi ceremony",
     "baraat", "varmala", "pheras", "vidai", "grah pravesh", "namkaran",
     "mundan", "thread ceremony", "engagement ring", "honeymoon"][:50]],

    # Round 3: Tech & modern life
    [f"{a}" for a in ["Phone not charging", "Laptop slow", "WiFi not working",
     "data recharge", "screen cracked", "app crash", "battery draining",
     "storage full", "virus alert", "OTP not received", "bank app error",
     "UPI failed", "payment pending", "refund delay", "order cancelled",
     "delivery late", "wrong item", "return policy", "exchange process",
     "Flipkart sale", "Amazon offer", "Myntra haul", "Meesho order",
     "Instagram reels", "YouTube channel", "Twitter trend", "LinkedIn post",
     "Facebook memory", "WhatsApp group", "Telegram channel", "Discord server",
     "Zoom bomb", "Google Meet", "Teams meeting", "screen share",
     "password reset", "account hacked", "privacy setting", "backup data"][:40]],

    # Round 4: Health, food & emotions
    [f"{a}" for a in ["fever", "cold", "cough", "headache", "stomach pain",
     "toothache", "back pain", "knee pain", "eye strain", "skin rash",
     "blood test", "X-ray report", "doctor appointment", "hospital visit",
     "insurance claim", "medicine reminder", "diet plan", "weight loss",
     "protein shake", "gym workout", "yoga asanas", "meditation",
     "stress relief", "anxiety", "overthinking", "sad mood", "lonely",
     "missing someone", "old memories", "regret", "forgiveness", "hope",
     "motivation", "success story", "failure lesson", "dream big",
     "chai biscuit", "pav bhaji", "dosa sambar", "idli chutney",
     "golgappa", "chaat papdi", "biryani", "kebab", "butter chicken",
     "dal makhani", "roti sabzi", "aloo paratha", "paneer"][:53]],

    # Round 5: Work, career & money
    [f"{a}" for a in ["resume writing", "job search", "interview tips", "salary negotiation",
     "promotion", "appraisal", "bonus", "increment", "resignation", "notice period",
     "joining bonus", "relocation", "WFO vs WFH", "side hustle", "freelancing",
     "Upwork profile", "Fiverr gig", "client call", "project proposal", "invoice",
     "payment pending", "tax filing", "ITR", "form 16", "PF withdrawal",
     "bank loan", "credit card", "Cibil score", "interest rate", "down payment",
     "rent agreement", "lease", "security deposit", "broker fee", "electricity bill",
     "water bill", "maintenance", "society rules", "parking space", "visitor entry"][:40]]
]

def gen_batch(topics_batch):
    topics_str = ", ".join(topics_batch[:8])
    prompt = f"""Generate 6 short Hinglish conversations about these topics: {topics_str}

Each: USER asks in casual Hinglish (Hindi+English mix), ASSISTANT replies naturally.

Examples of Hinglish:
- "Yaar, kal ka match dekh liya kya?" → "Haan bhai, Virat ne kamaal kar diya!"
- "Chai peetey hain?" → "Haan, ek cutting chai ho jaye!"

Use words like: yaar, bhai, haan, nahi, kya, kaise, theek, arre

Output JSON array: [{{"user":"...","assistant":"..."}}, ...]"""

    try:
        resp = session.post("https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800, "temperature": 0.9}, timeout=45)
        if resp.status_code != 200: return []
        content = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r'\[.*\]', content, re.DOTALL)
        try: return json.loads(m.group()) if m else json.loads(content)
        except: return []
    except: return []

count = sum(1 for _ in open("data/synthetic/hinglish_conversations.jsonl")) if os.path.exists("data/synthetic/hinglish_conversations.jsonl") else 0
print(f"Starting with {count} existing conversations")

for round_idx, topics in enumerate(ROUND_TOPICS):
    print(f"\nRound {round_idx+1}/{len(ROUND_TOPICS)}: {len(topics)} topics")
    round_start = time.time()
    new_convos = []
    
    with ThreadPoolExecutor(max_workers=12) as ex:
        batch_size = 8
        batches = [topics[i:i+batch_size] for i in range(0, len(topics), batch_size)]
        futures = {ex.submit(gen_batch, b): b for b in batches}
        
        for i, f in enumerate(as_completed(futures)):
            new_convos.extend(f.result())
            if (i+1) % 3 == 0:
                print(f"  {i+1}/{len(batches)} batches | {len(new_convos)} new convos | {time.time()-round_start:.0f}s")
    
    # Append to dataset
    with open("data/synthetic/hinglish_conversations.jsonl", "a") as f:
        for c in new_convos:
            if isinstance(c, dict) and "user" in c and "assistant" in c:
                f.write(json.dumps(c) + "\n")
    
    count += len(new_convos)
    print(f"  Round done: {len(new_convos)} new | Total: {count} | Time: {time.time()-round_start:.0f}s")
    
    # Push to GitHub after each round
    os.system("git add data/synthetic/hinglish_conversations.jsonl && git commit -m 'Add Hinglish data' && git push 2>/dev/null")

print(f"\n=== FINAL: {count} total conversations ===")

# Show samples
print("\n=== Random Samples ===")
import random
with open("data/synthetic/hinglish_conversations.jsonl") as f:
    lines = [json.loads(l) for l in f if l.strip()]
random.shuffle(lines)
for i, c in enumerate(lines[:5]):
    print(f"\n{i+1}. U: {c['user'][:100]}")
    print(f"   A: {c['assistant'][:120]}")
