#!/usr/bin/env python3
"""Build the v4 Hinglish-first SFT set from the v3 conversation corpus.

The v3 SFT used ~35K rows from a 436K-row machine-translated corpus. The result
was a model that could answer a math question and then drift into Thai/Arabic
junk, and that sometimes gave human-medicine advice for crops. v4 starts from a
stronger base (Qwen3.5-2B) and a cleaner, language-balanced set.

Filters
-------
* keep conversations with at least one user + one assistant turn
* reject assistant turns containing CJK/Arabic/Thai script (base contamination)
* reject echoes, empty turns, extreme lengths, and obvious PII
* reject a small abuse/medical-prescription blocklist
* dedupe by normalized first user + assistant text
* balance: ~70% Hinglish/mixed, ~20% Devanagari, ~10% English
* prepend one consistent system prompt (the same one used at inference)
* append a small identity/safety seed set (no BharatBench prompts)

Output
------
  data/v4/sft/train.jsonl
  data/v4/sft/valid.jsonl
  data/v4/sft/stats.json

Each row is `{"messages": [...], "lang": "hinglish|deva|english", "source": "..."}`.

Usage
-----
  venv/bin/python scripts/build_v4_sft.py \
    --corpus "/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl" \
    --out-dir data/v4/sft --max-rows 30000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,23}")
CJK_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")

HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kaise", "nahi", "nahin", "bhi", "yeh", "toh",
    "chahiye", "bilkul", "bahut", "aap", "main", "mein", "hum", "kar",
    "karo", "karna", "hoga", "raha", "rahi", "rahe", "bhai", "yaar", "kuch",
    "sab", "thik", "theek", "acha", "accha", "haan", "ji", "matlab", "samajh",
    "dekho", "batao", "chalo", "dena", "lena", "wala", "wali", "log", "ghar",
    "kaam", "paisa", "paise", "sarkar", "yojana", "kisan", "kheti", "fasal",
}

# A deliberately small blocklist. The goal is to remove clearly unsafe rows,
# not to build a content moderator.
ABUSE_RE = re.compile(
    r"\b(kill|murder|rape|suicide|terrorist|bomb|acid attack|lynch)\b", re.I
)
PRESCRIPTION_RE = re.compile(
    r"\b(paracetamol|azithromycin|amoxicillin|ibuprofen|antibiotic)\b"
    r".{0,40}\b(do|de|lein|lo|khayein|khilayein|mg|tablet)\b",
    re.I,
)
# Web-scrape artifacts seen in the v3 corpus. These are not conversations.
WEB_ARTIFACT_RE = re.compile(
    r"(sEligibility|Application ProcessDocuments|Frequently Asked QuestionsSources"
    r"|QuestionsSources|under this scheme \?|What is the age limit for the students \?"
    r"|Checklist - Ready to Deliver|NowCheck|Check Eligibility|Apply Now"
    r"|SchemeAre you sure|Scheme ke liye: Now)",
    re.I,
)

SYSTEM_PROMPT = (
    "You are Bharat-Tiny-LLM, a helpful offline assistant for India. "
    "Reply in the same language and script the user used: Hinglish, Hindi "
    "(Devanagari), or English. Be concise and practical. Never invent live "
    "prices, weather, dates, scores, or government scheme details; say you "
    "cannot check live data and tell the user where to verify. For health "
    "questions, do not diagnose or prescribe; recommend a doctor, ASHA worker, "
    "or 108 for emergencies. For crop questions, prefer safe cultural/IPM steps "
    "and refer to a local KVK or agriculture officer."
)

IDENTITY_ROWS = [
    ("Tum kaun ho?", "Main Bharat-Tiny-LLM hoon, ek offline Hindi/Hinglish assistant. Mujhe eulogik ne Apache-2.0 license ke under banaya hai."),
    ("Tumhara naam kya hai?", "Mera naam Bharat-Tiny-LLM hai. Main Hindi, Hinglish aur English mein madad kar sakta hoon."),
    ("Kya tum GPT ho?", "Nahi, main Bharat-Tiny-LLM hoon. Main Qwen3.5 base par bana ek alag open model hoon."),
    ("Kya tum online ho?", "Main offline GGUF ke roop mein chal sakta hoon. Jab internet nahi hota, tab bhi basic madad kar sakta hoon."),
    ("Tumhe kisne banaya?", "Mujhe eulogik ne banaya hai. Weights Apache-2.0 hain aur GitHub par open source hain."),
    ("Mera data kahan jata hai?", "Offline GGUF mode mein aapka data device se bahar nahi jata. Hosted demo API use karta hai, to wahan data server tak jata hai."),
]

SAFETY_ROWS = [
    (
        "Mere tamatar ke patton par safed daag aa rahe hain, kya karu?",
        "Bhai, safed daag ki wajah alag-alag ho sakti hai — powdery mildew, whitefly, ya paani ki samasya. Ek saaf photo bhejiye aur patte ke neeche bhi dikhaiye. Tab tak: 1) affected patte hata kar safe dispose karein, 2) paani subah dein taaki patte sookh jayein, 3) paudhon ke beech hawa aane dein. Koi bhi dawai label ke bina ya KVK/agri officer se poochhe bina na chhidkein.",
    ),
    (
        "Mere aloo ke patte mud rahe hain, kya karu?",
        "Aloo ke patte mudna pani, garmi, ya kisi bimari ka sign ho sakta hai. Photo aur mitti ki nami ki jankari bhejiye. Filhaal extra paani na dein, khet mein drainage check karein, aur affected paudhe alag karein. Sahi wajah ke liye apne nazdiki KVK ya agriculture officer se sampark karein.",
    ),
    (
        "Bukhar mein paracetamol dena chahiye?",
        "Main dose ya dawai nahi bata sakta. Bukhar ho to doctor ya ASHA worker se poochein, khaas kar bachchon, pregnant mahilaon, ya 3 din se zyada bukhar mein. Paani/pedialyte se hydration rakhein. Agar saans phool rahi ho, behoshi ho, daura ho, ya bukhar bahut tez ho to turant 108 par call karein ya nazdiki aspatal jayein.",
    ),
    (
        "Saanp ne kaata hai, kya karna chahiye?",
        "Turant 108 par call karein ya nazdiki aspatal lein. Kata hua ang ko hilayein kam, gehne hata dein, aur jitna ho sake shaant rakhein. Kaatne ki jagah ko kaatna, chusna, ya tight bandage baandhna nahi chahiye. Ambulance aane tak vyakti ko leta rakhein aur uski saans par nazar rakhein.",
    ),
]

DOMAINS = ("agriculture", "schemes", "education", "health", "daily")


def classify(text: str) -> str:
    latin = len(LATIN_RE.findall(text))
    deva = len(DEVA_RE.findall(text))
    alpha = latin + deva
    if alpha == 0:
        return "other"
    words = [w.lower() for w in WORD_RE.findall(text)]
    markers = sum(1 for w in words if w in HINGLISH_MARKERS)
    if latin / alpha >= 0.75 and markers == 0:
        return "english"
    if deva / alpha >= 0.85:
        return "deva"
    if latin / alpha >= 0.55 and markers >= 2:
        return "hinglish"
    return "mixed"


def clean_turn(text: str) -> str:
    text = text.strip()
    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = AADHAAR_RE.sub("[id]", text)
    return re.sub(r"\s+", " ", text).strip()


def bad_assistant(text: str) -> bool:
    if not text or len(text) < 25:
        return True
    if CJK_RE.search(text) or ARABIC_RE.search(text) or THAI_RE.search(text):
        return True
    if WEB_ARTIFACT_RE.search(text):
        return True
    if not re.search(r"[.!?।]", text) and len(text) < 120:
        return True
    if text.count("?") >= 3:
        return True
    if text.lower().strip() in {"ok", "okay", "thanks", "thank you", "hmm"}:
        return True
    if ABUSE_RE.search(text):
        return True
    if PRESCRIPTION_RE.search(text):
        return True
    return False


def echo_ratio(user: str, assistant: str) -> float:
    a = set(re.findall(r"\w+", user.lower()))
    b = set(re.findall(r"\w+", assistant.lower()))
    if not a:
        return 0.0
    return len(a & b) / len(a)


def fingerprint(user: str, assistant: str) -> str:
    norm = re.sub(r"\W+", " ", (user + " " + assistant).lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def iter_rows(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages")
            if not msgs or len(msgs) < 2:
                continue
            yield msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl")
    ap.add_argument("--out-dir", default="data/v4/sft")
    ap.add_argument("--max-rows", type=int, default=30000)
    ap.add_argument("--hinglish-share", type=float, default=0.70)
    ap.add_argument("--deva-share", type=float, default=0.20)
    ap.add_argument("--english-share", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")

    quotas = {
        "hinglish": int(args.max_rows * args.hinglish_share),
        "mixed": int(args.max_rows * args.hinglish_share),
        "deva": int(args.max_rows * args.deva_share),
        "english": int(args.max_rows * args.english_share),
    }
    # hinglish + mixed share one bucket
    quotas["hinglish_mixed"] = quotas["hinglish"] + quotas["mixed"]

    seen: set[str] = set()
    counts = Counter()
    kept = Counter()
    rejects = Counter()
    rows: list[dict] = []

    for msgs in iter_rows(corpus):
        if len(rows) >= args.max_rows:
            break
        cleaned = []
        ok = True
        for m in msgs:
            role = m.get("role")
            content = clean_turn(m.get("content", ""))
            if role not in ("user", "assistant"):
                continue
            if len(content) < 2 or len(content) > 4000:
                ok = False
                break
            cleaned.append({"role": role, "content": content})
        if not ok or len(cleaned) < 2:
            rejects["structure"] += 1
            continue
        # Require alternating user/assistant with a user first.
        if cleaned[0]["role"] != "user" or not any(m["role"] == "assistant" for m in cleaned):
            rejects["structure"] += 1
            continue
        user = next((m["content"] for m in cleaned if m["role"] == "user"), "")
        assistant = next((m["content"] for m in cleaned if m["role"] == "assistant"), "")
        if bad_assistant(assistant):
            rejects["assistant_quality"] += 1
            continue
        if user.lower().strip() == assistant.lower().strip():
            rejects["echo"] += 1
            continue
        if assistant.lower().strip() in user.lower() or user.lower().strip() in assistant.lower():
            rejects["echo"] += 1
            continue
        if echo_ratio(user, assistant) > 0.7:
            rejects["echo"] += 1
            continue

        lang = classify(user + " " + assistant)
        if lang not in ("hinglish", "mixed", "deva", "english"):
            rejects["language"] += 1
            continue
        counts[lang] += 1

        bucket = "hinglish_mixed" if lang in ("hinglish", "mixed") else lang
        if kept[bucket] >= quotas.get(bucket, 0):
            rejects["quota"] += 1
            continue

        fp = fingerprint(user, assistant)
        if fp in seen:
            rejects["duplicate"] += 1
            continue
        seen.add(fp)

        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + cleaned,
                     "lang": lang, "source": "train_gold_v3"})
        kept[bucket] += 1

    # Seed identity/safety rows. These are not from BharatBench and do not
    # duplicate its prompts; they fix the brand identity and the two worst
    # failure modes seen in the base model.
    for user, assistant in IDENTITY_ROWS + SAFETY_ROWS:
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "lang": "hinglish" if LATIN_RE.search(user) else "deva",
            "source": "seed_identity_safety",
        })
        kept["seed"] += 1

    random.shuffle(rows)
    n_valid = max(200, int(len(rows) * 0.02))
    valid, train = rows[:n_valid], rows[n_valid:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "valid.jsonl").open("w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "corpus": str(corpus),
        "max_rows": args.max_rows,
        "rows": len(rows),
        "train": len(train),
        "valid": len(valid),
        "seen_languages": dict(counts),
        "kept": dict(kept),
        "rejects": dict(rejects),
        "system_prompt": SYSTEM_PROMPT,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
