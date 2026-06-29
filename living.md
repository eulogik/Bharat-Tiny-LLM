# Bharat-Tiny-LLM: Living Document
**Last Updated:** Week 1, Day 1 (Session End)
**Status:** ✅ Production Model Complete — Qwen2.5-1.5B + Hinglish LoRA

---

## Project Overview

**What:** India's first native edge AI for Indian languages
**Why:** Zero tiny Indic models exist. Everyone builds skyscrapers, no one builds bicycles.
**Target:** Model running on ₹8,000 phones
**Status:** ✅ Production model complete — Qwen2.5-1.5B fine-tuned on 177K Hinglish conversations, Q4 quantized to 828 MB

---

## Current Status

### Production Model: Bharat-Tiny-LLM v5

| Metric | Value |
|--------|-------|
| Base model | Qwen2.5-1.5B (multilingual, 29 languages) |
| Training data | 177K Hinglish conversations |
| Method | LoRA (16 layers, rank 8, alpha 16) |
| Training iters | 20,000 |
| Val loss | 1.168 (still decreasing) |
| Trainable params | ~3.5M (0.23% of 1.5B) |
| Peak memory | 4.87 GB |
| Training speed | ~2.2 it/s, ~160 tok/s |
| Total training time | ~5 hours |
| Q4 model size | 828 MB |
| Inference speed | ~20 tok/s on Mac Mini M4 |
| License | Apache 2.0 (base weights) |

### Why We Pivoted from SmolLM2-360M

SmolLM2 hit a hard ceiling after 50K LoRA iterations:
- Val loss plateaued at 1.32 (vs Qwen2.5's 1.168)
- English-only tokenizer: Hindi words tokenized as 3-5 subword tokens
- LoRA modifies only 0.3% (1M/361M) params — fundamentally limited
- Model produces grammatically plausible but semantically nonsensical Hinglish

Qwen2.5-1.5B advantages:
- Already has multilingual representations for 29 languages including Hindi
- 1.5B params means LoRA has more to work with
- Val loss still decreasing at 20K iters — more training headroom

### Training Data

| Source | Samples | Type |
|--------|---------|------|
| Hinglish-Everyday-Conversations-1M (HuggingFace) | 100,000 | Casual chit-chat |
| API-generated (OpenRouter) + augmented | 77,000 | Informative Q&A |
| **Total** | **177,000** | Mixed casual + informative |

---

## Hardware Setup

**Primary Development Machine:**
- Model: Mac Mini M4
- RAM: 16GB unified memory
- OS: macOS

**Software:**
- Python 3.14
- MLX 0.31.2 (Apple Silicon optimized)
- PyTorch 2.12.1
- transformers 5.12.1

---

## Project Structure

```
Brahmi/
├── .env                    # Credentials (gitignored)
├── .gitignore              # Git exclusions
├── living.md               # This document
├── IMPLEMENTATION_PLAN.md  # 32-week roadmap
├── BRAHMI_Build_GTM_Plan.md # Original vision doc
├── Strategic_Positioning.md # Market analysis
├── README.md               # Project overview with results
├── models/
│   ├── bharat-tiny-llm-qwen-q4/  # Final Q4 model (828 MB)
│   └── adapters/
│       ├── qwen_v1/              # Qwen2.5 LoRA adapter (20K iters)
│       └── prod_v4/              # SmolLM2 adapter (archived)
├── data/
│   ├── processed/
│   │   ├── train.jsonl           # 159K training conversations
│   │   └── valid.jsonl           # 17K validation conversations
│   └── synthetic/
│       └── hinglish_conversations.jsonl  # 3,858 API-generated
├── src/
│   └── train.py                  # Training orchestrator
├── scripts/
│   ├── mass_generate.py          # Parallel data generator
│   ├── generate_final.py         # Sequential generator
│   └── augment_data.py           # Data augmentation
└── configs/
    └── training.yaml             # Training hyperparameters
```

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Week 0 | Use SmolLM2-360M as base | Apache 2.0, 360M params, strong English |
| Week 0 | Mac Mini M4 only | $0 cloud cost, democratizes development |
| Week 0 | Warm-start training | Cannot train from scratch (4+ years) |
| Week 0 | Custom tokenizer | 64K vocab, Unigram, trained on 1.36GB Indic corpus |
| Week 1 | Switch to messages format | Better QA learning vs text format |
| Week 1 | Use OpenRouter free models | qwen/qwen3-8b for cheap data gen |
| Week 1 | **Pivot to Qwen2.5-1.5B** | SmolLM2 English-only base + LoRA fundamentally limited |
| Week 1 | Use base model (not instruct) | Avoid conflicting RLHF/DPO constraints |
| Week 1 | 16 LoRA layers on Qwen2.5 | More representational capacity than SmolLM2's 8 |

---

## Training Rounds

### SmolLM2-360M (Archived)

| Round | Iters | Data | Val Loss | Notes |
|-------|-------|------|----------|-------|
| poc_v1 | 300 | 227 | 1.03 | Text format |
| poc_v2 | 500 | 351 | 0.39 | Text format |
| poc_v3 | 800 | 351 | 0.38 | Messages format |
| prod_v1 | 1,200 | 561 | 0.39 | Messages |
| prod_v2 | 2,000 | 1,111 | 0.81 | Messages |
| prod_v3 | 3,000 | 3,858 | 1.83 | Messages |
| prod_v4 | 50,000 | 3,858 | 1.32 | **Plateaued** |

### Qwen2.5-1.5B (Production)

| Round | Iters | Data | Val Loss | Notes |
|-------|-------|------|----------|-------|
| qwen_v1 | 20,000 | 177K | 1.168 | **Still decreasing** |

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| OpenRouter rate limits | ⚠️ Known | Free tier only, no credits |
| Val loss still decreasing | ⚠️ Opportunity | More training could improve further |
| Biryani recipe includes "dough" | ⚠️ Quality | More diverse food data needed |
| Occasional English sentences | ⚠️ Quality | More Hinglish-only training data |

---

## Next Actions

### Short term (Next session)
1. Train Qwen2.5-1.5B longer (50K+ iters) — val loss still decreasing
2. Test with higher LoRA rank (16 or 32) for more capacity
3. Add more diverse Hinglish data (food, travel, tech support)
4. Benchmark on Raspberry Pi 5 / Android
5. Create demo: "Chat in Hinglish on ₹8,000 phone"

### Medium term (Next month)
1. Implement DPO/RLHF for quality improvement
2. Add 8 more languages (Tamil, Telugu, Bengali, Marathi)
3. Create BharatTiny-Bench evaluation suite
4. Release on HuggingFace
5. Create web demo with Gradio

### Long term (Next quarter)
1. Add CKH (Cultural Knowledge Hypernetwork)
2. Add EFP (Edge Federated Personalization)
3. Multimodal (VLM for documents)
4. BRAHMI Studio — no-code fine-tuning

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| Week 0 | Initial setup, SmolLM2-360M downloaded | Eulogik |
| Week 0 | Custom tokenizer trained (64K, Unigram) | Eulogik |
| Week 0 | poc_v1–poc_v3 trained (Quick experiments) | Eulogik |
| Week 0 | prod_v1–prod_v2 trained | Eulogik |
| Week 1 | Data augmentation (3,858 conversations) | Eulogik |
| Week 1 | prod_v3–prod_v4 trained, SmolLM2 plateaued | Eulogik |
| Week 1 | Downloaded Hinglish-Everyday-1M (1M samples) | Eulogik |
| Week 1 | Built 177K conversation training set | Eulogik |
| Week 1 | **Pivoted to Qwen2.5-1.5B** | Eulogik |
| Week 1 | qwen_v1 trained (20K iters, val loss 1.168) | Eulogik |
| Week 1 | Q4 quantized fused model (828 MB, 20 tok/s) | Eulogik |
| Week 1 | Final push — **Production model complete** | Eulogik |

---

*This document is updated after every significant milestone. Keep it current.*
