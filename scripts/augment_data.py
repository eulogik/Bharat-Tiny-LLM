#!/usr/bin/env python3
"""Data augmentation for Hinglish conversations"""
import json, random, re, copy

with open("data/synthetic/hinglish_conversations.jsonl") as f:
    samples = [json.loads(l) for l in f if l.strip()]

print(f"Original: {len(samples)}")

# Hinglish synonyms for augmentation
SYNONYM_SWAPS = {
    "kya": ["kya", "kyu", "kaisa"],
    "kaise": ["kaise", "kis tarah", "kya tareeke se"],
    "bahut": ["bahut", "kafi", "zyada", "beshumar"],
    "accha": ["accha", "theek", "sahi", "badhiya"],
    "chai": ["chai", "cha", "cutting chai"],
    "karo": ["karo", "karein", "kijiye", "kar"],
    "nahi": ["nahi", "na", "mat"],
    "haan": ["haan", "haji", "hmm", "bilkul"],
    "theek": ["theek", "sahi", "accha", "okay"],
    "bhai": ["bhai", "bhaai", "dost", "yaar", "mitra"],
    "yaar": ["yaar", "dost", "bhai", "mere bhai"],
    "dekho": ["dekho", "dekh", "dekhiye", "check karo"],
    "bolo": ["bolo", "bol", "batao", "bata"],
    "aaja": ["aaja", "aaiye", "chalo", "aao"],
    "jaa": ["jaa", "jaaiye", "chale jao"],
    "aap": ["aap", "tum", "tu"],
    "mera": ["mera", "apna", "hamara"],
    "kuch": ["kuch", "thoda", "koyi"],
    "sab": ["sab", "saara", "poora"],
}

ABBREVIATIONS = {
    "kya hai": "kya h",
    "nahi hai": "nahi h",
    "kaisa hai": "kaisa h",
    "kya karun": "kya karu",
    "theek hai": "thik h",
    "accha hai": "accha h",
}

def augment_sample(s, intensity=1):
    variants = []
    user = s["user"]
    assistant = s["assistant"]
    
    # V1: Swap Hinglish words with synonyms
    for _ in range(intensity):
        u = user
        a = assistant
        words_u = u.split()
        words_a = a.split()
        
        # 30% chance to swap a word
        for j, w in enumerate(words_u):
            if w.lower() in SYNONYM_SWAPS and random.random() < 0.3:
                words_u[j] = random.choice(SYNONYM_SWAPS[w.lower()])
        for j, w in enumerate(words_a):
            if w.lower() in SYNONYM_SWAPS and random.random() < 0.3:
                words_a[j] = random.choice(SYNONYM_SWAPS[w.lower()])
        
        u_new = " ".join(words_u)
        a_new = " ".join(words_a)
        
        if (u_new, a_new) != (user, assistant):
            variants.append({"user": u_new, "assistant": a_new, "augmented": True})
    
    # V2: Apply abbreviations
    u = user
    a = assistant
    for orig, abbr in ABBREVIATIONS.items():
        if random.random() < 0.4:
            u = u.replace(orig, abbr)
            a = a.replace(orig, abbr)
    if (u, a) != (user, assistant) and (u, a) not in [(v["user"], v["assistant"]) for v in variants]:
        variants.append({"user": u, "assistant": a, "augmented": True})
    
    # V3: Change sentence beginning
    beginnings = ["Arre ", "Haan ", "Abe ", "Oye ", "", "Chalo ", "Sun "]
    if random.random() < 0.3:
        u = user
        # Don't double-prefix
        if not any(user.startswith(b.strip()) for b in beginnings if b.strip()):
            u = random.choice(beginnings) + user[0].lower() + user[1:]
            variants.append({"user": u, "assistant": a, "augmented": True})
    
    return variants

random.seed(42)
augmented = []
for s in samples:
    augmented.extend(augment_sample(s, intensity=2))

print(f"Augmented: {len(augmented)}")

# Dedup
seen = set()
all_data = []
for s in samples + augmented:
    key = (s.get("user",""), s.get("assistant",""))
    if key not in seen:
        seen.add(key)
        all_data.append(s)

print(f"Total unique (original + augmented): {len(all_data)}")

# Random samples
random.shuffle(all_data)
print("\n=== Samples ===")
for i, c in enumerate(all_data[:5]):
    print(f"\n{i+1}. U: {c['user'][:100]}")
    print(f"   A: {c['assistant'][:100]}")

# Save
with open("data/synthetic/hinglish_conversations.jsonl", "w") as f:
    for s in all_data:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"\nSaved to file: {len(all_data)} conversations")
