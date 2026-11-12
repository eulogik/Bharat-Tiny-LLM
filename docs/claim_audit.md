# Claim audit — Bharat-Tiny-LLM v3

**Date:** 2026-11-11
**Status:** pre-launch correction. The claims marked ❌ must not be repeated in
the README, HF card, PyPI, demo Space, talks, or launch posts.
**Scope:** README, `docs/benchmarks_v2.md`, `docs/paper_arxiv.md`,
`docs/HF_MODEL_CARD.md`, `packages/pyproject.toml`, demo Space, and the public
Hub/PyPI pages they generate.

---

## TL;DR

1. **The headline bits/char result is invalid.** The benchmark shifted labels
   twice, so it scored token *i+2* instead of *i+1*. The reported
   `11.90 → 8.73 bpc (−26.6%)` is an artifact. With correct alignment the same
   models score **1.20 bpc (base) vs 1.71 bpc (v3)** on a 60-window smoke run —
   v3 is **42.8% worse**, not 26.6% better. A separate 305,857-character
   llama.cpp/Q4 run agrees in direction: **1.51 vs 1.72 bpc (v3 14.2% worse)**.
2. **The 56% GSM8K-Hindi number is already retracted** in the
   internal v3.5 decision record, but it is still in the README and
   `docs/benchmarks_v2.md`. Pinned protocols give 2% (torch/fp16) and 29%
   (llama.cpp/Q4) on n=100. The 56.2% was a 9/16 ad-hoc run with no committed
   protocol.
3. **Tokenizer fertility is real and should be the surviving quantitative
   claim.** v3 uses **36.87% fewer tokens on Devanagari** and **17.72% fewer on
   mixed text** than Qwen3-1.7B, measured on 9.68M held-out characters. The new
   Hinglish injection on Qwen3.5-2B gives **16.56% fewer tokens on held-out
   Hinglish** and **7.41% fewer overall**, with no English regression (0.21%
   savings). This is a tokenizer metric, not a quality win.
4. **"Runs fully offline" is true for the GGUF weights and false for the hosted
   demo.** The Space calls the Featherless API. Qualify every offline claim by
   artifact.
5. **The HF model card and PyPI package still describe v2** (1.5B, 880 MB, MLX,
   36.5%). They are not v3 cards. Either label them legacy or replace them.

---

## Claim-by-claim

| # | Claim | Where it appears | Evidence | Verdict | Action |
|---:|---|---|---|---|---|
| 1 | "GSM8K-Hindi 56%+" | README table; `docs/benchmarks_v2.md` | the internal v3.5 decision record: 9/16 ad-hoc run, protocol not committed; pinned n=100 gives 2% torch/fp16 and 29% llama.cpp/Q4; 56.2% "RETRACTED as irreproducible" | ❌ **False** | Delete from README and benchmarks. Never use "56%" in any public artifact. |
| 2 | "Bits/char 11.90 → 8.73 (−26.6%)" | README; paper abstract; Space line 94; `benchmarks_v2.md` | `scripts/bench_ppl_torch.py` double-shifts labels; corrected run: base 1.1973, v3 1.7094; llama.cpp/Q4 on 305,857 chars: base 1.51, v3 1.72 | ❌ **Invalid** | Delete the number. Replace with tokenizer fertility, or with the corrected bpc that shows no win. |
| 3 | "Devanagari token compression 36.8%" | README; paper; Space | `docs/bench_fertility_v3.json`: 1,042.9 → 658.3 tok/1000 chars on 3.36M Devanagari chars = −36.87% | ✅ **True** (tokenizer only) | Keep, but label "tokenizer fertility, not model quality." |
| 4 | "Mixed corpus savings 18.0%" | README; `benchmarks_v2.md` | `docs/bench_fertility_v3.json`: 586.5 → 482.6 tok/1000 chars on 9.68M chars = −17.72% | ✅ **True** (tokenizer only) | Keep with the same label. |
| 5 | "Runs fully offline" | README subtitle; HF card; demo | GGUF weights run offline; Space sends chats to Featherless API (its own FAQ says so) | ⚠️ **Half true** | Say "GGUF weights run offline; the hosted demo uses an API." |
| 6 | "1.0 GB GGUF" | README; Space | `brahmi-v3-q4_k_m.gguf` = 1.0 GB on disk | ✅ **True** | Keep. |
| 7 | "1.7B params" | README; Space | Qwen3-1.7B config: 1.72B | ✅ **True** | Keep. |
| 8 | "Apache-2.0" | README; HF card; PyPI | `LICENSE` is Apache-2.0; Qwen3-1.7B-Base is Apache-2.0 | ✅ **True** | Keep. |
| 9 | "33.8% token compression" | `packages/pyproject.toml` description | No artifact supports 33.8%. v3 is 36.87% Devanagari / 17.72% mixed; v2 is 36.5% / 18.0% | ❌ **Wrong number** | Replace with a verified figure, or drop the number from the package description. |
| 10 | "880 MB, 1.5B, MLX, 36.5%" | `docs/HF_MODEL_CARD.md`; `packages/README.md`; PyPI | These are v2 properties. v3 is 1.7B, 1.0 GB Q4 GGUF | ⚠️ **Stale** | Label the card/package "v2 legacy" or publish a real v3 card. |
| 11 | "v3.3 conversational: 370 rows, 46 steps, ~4 min" | README note | Training log/run artifacts not re-verified in this audit | ⚠️ **Unverified** | Verify from the training log before repeating; otherwise drop the step count. |
| 12 | "GSM8K-Hindi 29% (llama.cpp/Q4, n=100)" | Paper; decision log | `docs/bench_gsm_hi_v3_llama.json`: 29/100, 4-shot CoT greedy exact-match | ✅ **True with caveat** | Always state n=100, 4-shot, greedy, Q4, and the 2% torch/fp16 result. Never present as a single "accuracy." |
| 13 | "Qwen3 tokenizer does little better at 0.96 tokens/char" | Paper abstract | Measured Qwen3-1.7B: 1,042.9 tok/1000 Devanagari chars = **1.043 tokens/char**. 0.96 is Qwen2.5's *chars per token*, a units inversion | ❌ **Wrong** | Fix to 1.04 tokens/char. |
| 14 | "v3 is stronger at Hindi language modeling than base" | Space line 94; README implication | Corrected bpc says v3 is worse (1.71 vs 1.20 smoke; 1.72 vs 1.51 llama.cpp) | ❌ **False** | Delete. Replace with "v3 is a compact chat/math variant; it is not a better language model." |
| 15 | "v3.3 GSM 35%" | Space; decision log | `docs/bench_gsm_hi_v33_llama.json`: 35/100 pinned llama.cpp/Q4 | ✅ **True with caveat** | Keep protocol and n; note the 35% is the chat variant, not v3. |
| 16 | "SFT on Hindi math + QA" | README; Space | Training configs/logs exist; exact row count (35K) not re-verified | ⚠️ **Plausible, unverified** | Verify row count from the training file before launch. |
| 17 | "EvolucentAI" identity in v3.3 outputs | v3.3 chat samples | Conflicts with the eulogik brand; noted in the v4 plan | ❌ **Brand bug** | Fix in the next SFT; do not ship a model that names the wrong lab. |

---

## Why the bits/char number was wrong

The old benchmark did this:

```python
# scripts/bench_ppl_torch.py (old)
ids = torch.tensor([chunk[:-1]])
labels = torch.tensor([chunk[1:]])
out = model(ids, labels=labels)
```

Hugging Face causal-LM heads shift labels internally:

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

So the old code compared `logits[i]` with `labels[i+1] = chunk[i+2]`. It was
predicting two tokens ahead, not one. The fix is:

```python
ids = torch.tensor([chunk])
out = model(ids, labels=ids)   # one internal shift, correct
```

A direct check on one Hindi sentence:

| Alignment | Loss (nats/token) | PPL |
|---|---:|---:|
| Correct (`model(ids, labels=ids)`) | 0.6285 | 1.87 |
| Double-shift (`model(ids[:-1], labels=ids[1:])`) | 18.5488 | 113,663,814 |

That is the entire source of the paper's `1,703,571` base PPL and the
`331,235` v3 PPL. The corrected benchmark is `scripts/bench_bpc.py`.

### Corrected numbers

Protocol: fixed 1,024-character windows, one forward pass per window, correct
causal alignment, bits/char and bits/byte, bootstrap CI over windows.
Smoke run: 60 windows / 58,210 characters from
`data/processed/valid_gold.jsonl`; MPS fp16.

| Model | tok/1000 chars | bits/char | 95% CI | bits/byte | PPL |
|---|---:|---:|---|---:|---:|
| Qwen3-1.7B-Base | 600.94 | **1.1973** | 1.1508–1.2400 | 0.6891 | 3.98 |
| Bharat-Tiny-LLM v3 | 492.46 | **1.7094** | 1.6281–1.7914 | 0.9838 | 11.09 |

The confidence intervals do not overlap. By slice (same windows):

| Slice | Base bpc | v3 bpc | v3 vs base |
|---|---:|---:|---:|
| Hinglish (30 windows) | 1.2226 | 1.4697 | +20.2% worse |
| Devanagari (28 windows) | 1.1910 | 1.9629 | +64.8% worse |
| Mixed (2 windows) | 0.8076 | 1.3306 | +64.8% worse |

Independent llama.cpp/Q4 check (305,857 held-out chars, from the earlier
measurement): base PPL 5.89 at 589.5 tok/1000 chars = **1.51 bpc**; v3 PPL
11.77 at 483.8 tok/1000 chars = **1.72 bpc**. Direction is the same: v3 is
worse in bits/char despite using fewer tokens.

**Conclusion:** injection bought token count, not information density. The
SFT stage degraded the base model's next-token prediction. This is why the v4
plan gates on task success and naturalness, not on token compression alone.

---

## The survivable story

What is actually true and worth saying:

> Bharat-Tiny-LLM v3 is a 1.7B Apache-2.0 Hindi/Hinglish/English model that
> ships as a 1.0 GB GGUF. A 300-token Devanagari injection cuts token count by
> **36.9% on Devanagari** and **17.7% on mixed Hindi/Hinglish text** versus
> Qwen3-1.7B, measured on 9.68M held-out characters. The model is a compact
> chat/math variant, not a frontier reasoning model: on pinned GSM8K-Hindi
> (4-shot CoT, greedy, n=100) it scores **29% under llama.cpp/Q4** and **2%
> under torch/fp16**, and the earlier 56% figure is retracted. The GGUF runs
> offline; the hosted demo uses an API.

What must not be said:

- "56% GSM8K-Hindi"
- "26.6% fewer bits/char"
- "stronger Hindi language modeling than base"
- "100% offline" without naming the artifact
- "33.8% compression"

---

## Immediate edit list

| File | Edit |
|---|---|
| `README.md` | Remove 56% and 26.6%; replace the capability table with tokenizer fertility + honest quality; label the demo as API-hosted; fix the v3.3 note if unverified. |
| `docs/benchmarks_v2.md` | Add a correction banner at the top; remove the v3 56%/26.6% rows; keep fertility. |
| `docs/paper_arxiv.md` | Fix the abstract (1.04 tok/char; no bpc win; 56% retracted; 29% with protocol). Do not resubmit. |
| `docs/HF_MODEL_CARD.md` | Replace with a v3 card or mark it explicitly `v2 legacy`. |
| `packages/pyproject.toml` | Remove "33.8%"; point the default at v3 GGUF or label the package v2. |
| `packages/bharat_tiny_llm/constants.py` | Default to the v3 GGUF, not v2 MLX. |
| `spaces/.../index.html` | Delete the "26.6% fewer bits/char" row; keep the API disclosure. |
| HF Hub v3 card | Mirror the corrected README; add the benchmark script link and the retraction note. |
| PyPI 0.4.0 | Publish after the code default is v3; describe v2 as legacy. |

---

## Verification commands

```bash
# corrected bits/char
venv/bin/python scripts/bench_bpc.py \
  --corpus data/processed/valid_gold.jsonl \
  --windows 60 --window-chars 1024 --seq 2048 --dtype fp16 \
  --model Qwen/Qwen3-1.7B-Base \
  --model "/Volumes/KIOXIA 1TB/brahmi-v3-merged" \
  --out docs/bench_bpc_smoke.json

# tokenizer fertility
cat docs/bench_fertility_v3.json

# retraction evidence: archived in the internal v3.5 decision record

# pinned GSM8K-Hindi
cat docs/bench_gsm_hi_v3_llama.json
cat docs/bench_gsm_hi_v3_torch.json
```

---

## Bottom line

The v3 line has one durable technical result — **Devanagari/Hinglish token
compression** — and one durable product result — **a 1.0 GB offline GGUF**.
It does not have a language-modeling or math breakthrough. The launch should
say that plainly, then show the v4 work that actually targets usability:
Hinglish-first injection, voice, RAG, and a public benchmark. Credibility is
the scarce resource; the corrected numbers are the only foundation for it.
