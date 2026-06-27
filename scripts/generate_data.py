#!/usr/bin/env python3
"""
Generate Hinglish training data via OpenRouter API
Uses parallel requests for speed
"""
import json, os, time, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API key
with open(".env") as f:
    for line in f:
        if line.startswith("OPENROUTER_API_KEY"):
            API_KEY = line.split("=", 1)[1].strip()
            break
    else:
        print("ERROR: OPENROUTER_API_KEY not found in .env")
        sys.exit(1)

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Session with retry
session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Topics
topics = [
    "cricket match", "Bollywood movie", "street food", "monsoon weather",
    "wedding prep", "college exams", "family dinner", "weekend plans",
    "IPL auction", "Diwali", "Holi", "auto rickshaw ride",
    "chai break", "morning routine", "train journey", "vegetable market",
    "phone recharge", "UPI payment", "movie review", "job interview",
    "road trip", "new phone", "exam results", "birthday party",
    "traffic jam", "festival shopping", "gym workout", "flight travel",
    "house renovation", "political rally",
]

def generate_batch(topic):
    """Generate 3 conversations for a topic."""
    prompt = f"""Generate 3 short Hinglish conversations about {topic}.

Each conversation: USER asks question in natural Hinglish, ASSISTANT replies naturally.

Hinglish = Hindi + English mixed naturally (yaar, bhai, haan, theek hai, kya, kaise).
Not translation. Real spoken Indian language.

Output JSON array: [{{"user":"...","assistant":"..."}}, ...]"""

    try:
        resp = session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "qwen/qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.9
            },
            timeout=30
        )
        
        if resp.status_code != 200:
            return []
        
        content = resp.json()["choices"][0]["message"]["content"]
        
        # Extract JSON array
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        try:
            return json.loads(content)
        except:
            return []
            
    except Exception:
        return []

print(f"Generating Hinglish data across {len(topics)} topics...")
start = time.time()

all_convos = []
os.makedirs("data/synthetic", exist_ok=True)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(generate_batch, t): t for t in topics}
    
    for i, future in enumerate(as_completed(futures)):
        topic = futures[future]
        convos = future.result()
        all_convos.extend(convos)
        
        # Save incrementally every 5 topics
        if (i + 1) % 5 == 0:
            with open("data/synthetic/hinglish_conversations.jsonl", "w") as f:
                for c in all_convos:
                    f.write(json.dumps(c) + "\n")
            print(f"  {i+1}/{len(topics)} topics done | {len(all_convos)} conversations | {time.time()-start:.0f}s elapsed")

# Final save
with open("data/synthetic/hinglish_conversations.jsonl", "w") as f:
    for c in all_convos:
        if isinstance(c, dict) and "user" in c and "assistant" in c:
            f.write(json.dumps(c) + "\n")

elapsed = time.time() - start
print(f"\n=== Done in {elapsed:.0f}s ===")
print(f"Total conversations: {len(all_convos)}")
print(f"Saved to: data/synthetic/hinglish_conversations.jsonl")

# Show samples
print("\n=== Samples ===")
with open("data/synthetic/hinglish_conversations.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        c = json.loads(line)
        print(f"\n--- {i+1} ---")
        print(f"User:     {c['user'][:120]}")
        print(f"Assist:   {c['assistant'][:150]}")
