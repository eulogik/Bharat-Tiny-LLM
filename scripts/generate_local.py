#!/usr/bin/env python3
"""Generate Hinglish training data using local Llama-3.2-3B via MLX"""
import json, os, re, time, random, sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

print("Loading model...")
m, t = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
sampler = make_sampler(temp=0.8)
print("Model loaded.")

TOPICS = [
    "chai biscuit, evening snacks", "biryani, restaurant, food", "phone battery, wifi, data recharge",
    "job interview, salary, resume", "train travel, ticket booking, station", "fever, cold, home remedy, doctor",
    "Diwali puja, Holi colors, festival", "Amazon sale, Flipkart, shopping", "exam stress, study tips, college",
    "UPI payment, bank, loan, saving", "wedding, family function, gift", "cricket match, movie, Netflix",
    "friend, relationship, dating", "gym, workout, yoga, diet", "car, bike, driving, traffic",
    "startup, business, funding", "house, rent, flatmate, landlord", "weather, rain, umbrella, cold",
    "birthday party, anniversary, celebration", "photo, Instagram, YouTube, reel",
]

def gen_conversations(topic, n=4):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate {n} short Hinglish conversations about "{topic}".

Each conversation has a USER asking in casual Hinglish (Hindi+English mix) and an ASSISTANT giving a helpful response.

Use words like: yaar, bhai, haan, nahi, kya, kaise, theek, arre, accha

Example conversations:
{{"user":"Yaar, phone ki battery bahut fast drain ho rahi hai, kya karun?","assistant":"Haan bhai, common problem hai. Pehle background apps check karo, screen brightness kam karo."}}
{{"user":"Chai peetey hain? Ek cutting chai ho jaye!","assistant":"Haan haan, bilkul! Adrak wali chai banau ya masala chai?"}}

Output ONLY a valid JSON array of objects with "user" and "assistant" keys. No other text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        r = generate(m, t, prompt=prompt, max_tokens=1500, sampler=sampler, verbose=False)
        m = re.search(r'\[.*\]', r, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            data = json.loads(r)
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict) and "user" in d and "assistant" in d
                and len(d["user"]) > 8 and len(d["assistant"]) > 10]
    except:
        return []

# Load existing count
count = sum(1 for _ in open("data/synthetic/hinglish_conversations.jsonl")) \
        if os.path.exists("data/synthetic/hinglish_conversations.jsonl") else 0
print(f"Starting with {count} conversations")

total_new = 0
start_time = time.time()
target = 2000

while total_new < target:
    topic = random.choice(TOPICS)
    batch_start = time.time()
    convos = gen_conversations(topic, 4)
    
    # Quick quality filter
    hinglish_markers = ["yaar", "bhai", "hai", "hoon", "kya", "kaise", "theek",
                       "nahi", "haan", "chai", "bahut", "accha", "mera", "karo",
                       "arre", "waah", "chalo", "dekho", "bolo", "aaja", "lekin"]
    quality = []
    for c in convos:
        text = (c["user"] + " " + c["assistant"]).lower()
        score = sum(1 for m in hinglish_markers if m in text)
        if score >= 1:
            quality.append(c)
    
    if quality:
        with open("data/synthetic/hinglish_conversations.jsonl", "a") as f:
            for c in quality:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    total_new += len(quality)
    elapsed = time.time() - start_time
    rate = total_new / elapsed * 3600 if elapsed > 0 else 0
    batch_time = time.time() - batch_start
    
    print(f"  topic={topic[:30]}.. got={len(quality)}/{len(convos)} total={total_new}/{target} batch={batch_time:.0f}s rate={rate:.0f}/hr")
    
    if total_new >= target:
        break

print(f"\n=== Done: {total_new} new conversations in {time.time()-start_time:.0f}s ===")
