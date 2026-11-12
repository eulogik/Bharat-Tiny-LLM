#!/usr/bin/env python3
"""Offline RAG over a small, source-linked India knowledge base.

No embeddings, no vector database: BM25 over ~25 curated chunks is enough for
the five domains v4 targets, runs on a phone-class CPU, and never needs a
network call. The retriever is deliberately simple so its failures are
inspectable.

Usage
-----
  # one question against a running llama-server
  .venv-v4/bin/python scripts/rag_ask.py \
    --server http://127.0.0.1:8096/v1 \
    --model Qwen3.5-2B-Q4_K_M.gguf \
    --query "PM Kisan ka status kaise check karein?"

  # compare with and without retrieval
  ... --no-rag

  # interactive
  ... 
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent.parent
DEFAULT_KB = HERE / "data/v4/kb/kb.jsonl"

# Local system layers: domain router + fixed critical-health protocols.
sys.path.insert(0, str(HERE / "scripts"))
try:
    from domain_prompt import system_for as route_system_for
except Exception:  # noqa: BLE001
    route_system_for = None
try:
    import safety_protocol
except Exception:  # noqa: BLE001
    safety_protocol = None
try:
    from text_quality import is_degenerate, loop_ratio
except Exception:  # noqa: BLE001
    is_degenerate = lambda text, **_: False  # type: ignore[assignment]
    loop_ratio = lambda text, **_: 0.0  # type: ignore[assignment]

SYSTEM_PROMPT = (
    "You are Bharat-Tiny-LLM, a helpful offline assistant for India. "
    "Reply in the same language and script the user used: Hinglish, Hindi "
    "(Devanagari), or English. Be concise and practical. Use the retrieved "
    "notes when they are relevant, but do not copy them blindly. Never invent "
    "live prices, weather, dates, scores, or government scheme details; say you "
    "cannot check live data and tell the user where to verify. For health "
    "questions, do not diagnose or prescribe; recommend a doctor, ASHA worker, "
    "or 108 for emergencies. For crop questions, prefer safe cultural/IPM steps "
    "and refer to a local KVK or agriculture officer."
)

WORD_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)

# Hinglish/Devanagari -> English KB vocabulary. This is deliberately small and
# domain-specific; it is not a general translator. Add terms as the KB grows.
EXPAND = {
    # crops and symptoms
    "tamatar": ["tomato"], "टमाटर": ["tomato"],
    "aloo": ["potato"], "आलू": ["potato"],
    "gehu": ["wheat"], "gehun": ["wheat"], "गेहूं": ["wheat"], "गेहूँ": ["wheat"],
    "dhan": ["rice", "paddy"], "धान": ["rice", "paddy"],
    "patte": ["leaf", "leaves"], "patta": ["leaf"], "patti": ["leaf"],
    "पत्ते": ["leaf", "leaves"], "पत्ता": ["leaf"],
    "peele": ["yellow", "yellowing"], "peela": ["yellow"], "पीले": ["yellow"],
    "safed": ["white"], "सफेद": ["white"],
    "daag": ["spot", "spots"], "दाग": ["spot"],
    "mud": ["curl", "curling"], "मुड़": ["curl"],
    "sookh": ["dry", "wilting"], "सूख": ["dry", "wilting"],
    "kheti": ["crop", "farming"], "खेती": ["crop", "farming"],
    "fasal": ["crop"], "फसल": ["crop"],
    "mitti": ["soil"], "मिट्टी": ["soil"],
    "khad": ["fertilizer"], "खाद": ["fertilizer"],
    "paani": ["water", "irrigation"], "पानी": ["water", "irrigation"],
    "keede": ["pest", "insect"], "कीड़े": ["pest"],
    "bimari": ["disease"], "बीमारी": ["disease"],
    # health
    "saanp": ["snake", "bite"], "सांप": ["snake"], "साँप": ["snake"],
    "kaata": ["bite"], "काटा": ["bite"],
    "dast": ["diarrhoea", "diarrhea"], "दस्त": ["diarrhoea"],
    "bukhar": ["fever"], "बुखार": ["fever"],
    "khansi": ["cough"], "खांसी": ["cough"],
    "sardi": ["cold"], "जुकाम": ["cold"],
    "pet": ["stomach", "abdomen"], "पेट": ["stomach"],
    "dard": ["pain"], "दर्द": ["pain"],
    "khoon": ["blood", "bleeding"], "खून": ["blood"],
    "jal": ["burn"], "jalan": ["burn"], "जलन": ["burn"],
    "haddi": ["fracture", "bone"], "हड्डी": ["fracture"],
    "dawai": ["medicine"], "दवाई": ["medicine"],
    "doctor": ["doctor"], "डॉक्टर": ["doctor"],
    "aspatal": ["hospital"], "अस्पताल": ["hospital"],
    # schemes and live data
    "yojana": ["scheme"], "योजना": ["scheme"],
    "kisan": ["farmer"], "किसान": ["farmer"],
    "kist": ["installment"], "किस्त": ["installment"],
    "bhav": ["price", "rate"], "भाव": ["price", "rate"],
    "rate": ["price", "rate"], "mandi": ["mandi", "market"], "मंडी": ["mandi"],
    "aavedan": ["apply", "application"], "आवेदन": ["apply"],
    "kagaz": ["documents", "papers"], "कागज": ["documents"],
    "paisa": ["money", "amount"], "पैसा": ["money"],
    "status": ["status"], "स्थिति": ["status"],
    "gaon": ["village"], "गांव": ["village"],
    "khet": ["field", "farm"], "खेत": ["field"],
}


def expand_query(text: str) -> str:
    terms = tokenize(text)
    extra: list[str] = []
    for t in terms:
        extra.extend(EXPAND.get(t, []))
    return " ".join(terms + extra)



def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.freqs = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter()
        for freq in self.freqs:
            df.update(freq.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query: str, doc_idx: int) -> float:
        q = tokenize(expand_query(query))
        score = 0.0
        freq = self.freqs[doc_idx]
        length = self.lengths[doc_idx] or 1
        for term in q:
            if term not in freq:
                continue
            idf = self.idf.get(term, 0.0)
            tf = freq[term]
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            )
        return score

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = [(self.score(query, i), i) for i in range(len(self.docs))]
        scores.sort(reverse=True)
        out = []
        for score, i in scores[:k]:
            if score <= 0:
                continue
            out.append({**self.docs[i], "score": round(score, 3)})
        return out


def load_kb(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["Retrieved notes (use only if relevant):"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] {c['title']} ({c['domain']})\n{c['text']}\nSource: {c['source']}")
    return "\n\n".join(lines)


def call(server: str, model: str | None, messages: list[dict], args,
         repeat_penalty: float | None = None) -> dict:
    payload = {
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if repeat_penalty is not None:
        payload["repeat_penalty"] = repeat_penalty
    if model:
        payload["model"] = model
    t0 = time.perf_counter()
    r = requests.post(server.rstrip("/") + "/chat/completions", json=payload, timeout=args.timeout)
    r.raise_for_status()
    d = r.json()
    elapsed = time.perf_counter() - t0
    msg = d["choices"][0]["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "total_s": round(elapsed, 3),
        "usage": d.get("usage", {}),
    }


def answer(question: str, bm25: BM25 | None, args) -> dict:
    # 1. Health is retrieval/protocol-only. No model call: a 2B model must not
    #    improvise diagnosis, medicine or first aid.
    if safety_protocol and not args.no_safety:
        domain0, _ = route_system_for(question) if route_system_for else ("daily", "")
        if domain0 == "health":
            text, meta = safety_protocol.health_answer(question)
            return {
                "content": text,
                "chunks": [],
                "safety_protocol": meta.get("protocol"),
                "health_layer": "protocol" if meta.get("protocol") else "generic",
                "routed_domain": "health",
                "total_s": 0.0,
                "usage": {},
            }

    # 2. Domain-specific system prompt + retrieval.
    if route_system_for and not args.no_router:
        domain, system = route_system_for(question)
    else:
        domain, system = None, SYSTEM_PROMPT

    chunks = bm25.search(question, args.top_k) if bm25 else []
    context = build_context(chunks)
    if context:
        system = system + "\n\n" + context
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    out = call(args.server, args.model, messages, args)
    if is_degenerate(out.get("content", "")):
        before = loop_ratio(out["content"])
        try:
            retry = call(args.server, args.model, messages, args, repeat_penalty=1.15)
            retry["loop_retry"] = True
            retry["loop_ratio_before"] = round(before, 3)
            out = retry
        except Exception:  # noqa: BLE001
            out["loop_retry"] = False
    out["loop_ratio"] = round(loop_ratio(out.get("content", "")), 3)
    if domain == "agriculture" and safety_protocol:
        lang = safety_protocol.detect_language(question)
        out["content"], replaced = safety_protocol.guard_crop(out["content"], lang, question)
        out["crop_guard"] = replaced
    out["chunks"] = chunks
    out["routed_domain"] = domain
    out["safety_protocol"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:8096/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kb", default=str(DEFAULT_KB))
    ap.add_argument("--query", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-router", action="store_true",
                    help="use the generic system prompt instead of per-domain routing")
    ap.add_argument("--no-safety", action="store_true",
                    help="do not use the fixed critical-health protocols (not recommended)")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    docs = load_kb(Path(args.kb))
    bm25 = None if args.no_rag else BM25(docs)
    print(f"KB: {len(docs)} chunks; RAG {'off' if args.no_rag else 'on'}", file=sys.stderr)

    def one(q: str) -> None:
        out = answer(q, bm25, args)
        print(f"\nQ: {q}")
        if out.get("routed_domain"):
            print(f"Domain: {out['routed_domain']}")
        if out.get("safety_protocol"):
            print(f"Safety protocol: {out['safety_protocol']} (fixed response, no model call)")
        print(f"A: {out['content']}")
        if out["chunks"]:
            print("Sources:")
            for c in out["chunks"]:
                print(f"  - {c['title']} ({c['source']}) score={c['score']}")
            if args.show_context:
                print(build_context(out["chunks"]))
        print(f"[{out['total_s']}s, {out['usage'].get('completion_tokens')} tokens]")

    if args.query:
        one(args.query)
        return

    print("RAG chat — Ctrl-C to exit.", file=sys.stderr)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            one(q)


if __name__ == "__main__":
    main()
