# Bharat-Tiny-LLM: Living Document
**Last Updated:** July 6, 2026
**Status:** ✅ Production Model v2 Complete — 76K iters, val loss 0.781 — Live on HuggingFace

---

## Project Overview

**What:** India's first native edge AI for Indian languages
**Why:** Zero tiny Indic models exist. Everyone builds skyscrapers, no one builds bicycles.
**Target:** Model running on ₹8,000 phones
**Status:** ✅ Production model v2 complete — Qwen2.5-1.5B fine-tuned on 376K Hinglish conversations, Q4 quantized to 828 MB, 57 tok/s. Live on HuggingFace under `eulogik` org.

---

## Current Status

### Production Model: Bharat-Tiny-LLM v6 (qwen_v2)

| Metric | Value |
|--------|-------|
| Base model | Qwen2.5-1.5B (multilingual, 29 languages) |
| Training data | 376K conversations (5 datasets) |
| Method | LoRA (16 layers, rank 8, alpha 16) |
| Training iters | 76,420 |
| Best val loss | **0.781** (@ 75K iters) |
| Data sources | HinglishConversations (202K), IndicVault (151K), HinglishInstruct (10K), cookGPT (6K), Yojana (7K) |
| Trainable params | 5.276M (0.342% of 1.5B) |
| Peak memory | 10.3 GB |
| Training speed | ~0.30 it/s, ~210 tok/s |
| Total training time | ~3.3 days |
| Q4 model size | 828 MB |
| Inference speed | **~57 tok/s** on Mac Mini M4 (3x faster than v5) |
| Best checkpoint | `models/adapters/qwen_v2/0075000_adapters.safetensors` |
| License | Apache 2.0 (base weights) |

### HuggingFace Release (July 6)

All assets live under the **eulogik** organization:

| Asset | URL | Format |
|-------|-----|--------|
| Q4 MLX Model | https://huggingface.co/eulogik/Bharat-Tiny-LLM | 828 MB, Q4 quantized, Apple MLX |
| PEFT Adapter | https://huggingface.co/eulogik/Bharat-Tiny-LLM-adapter | 20 MB, transformers+peft compatible |
| Gradio Demo | https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM | Live demo with eulogik branding |

**Positioning:** Transparent about Qwen2.5 base (collapsible "Base model details" section), brand-forward with "Bharat-Tiny-LLM" as the name. Full model card with story, badges, training curve, example outputs, limitations, and roadmap.

**Model card tags for SEO:** hinglish, hindi, indian-languages, edge-ai, mac-mini, lora, small-language-model, on-device-ai, apple-silicon, mlx, apache-2.0, lightweight, edge-deployment, low-resource, conversation, india, bharat

**Demo space:** Uses Dockerfile with Qwen2.5-1.5B pre-cached at build time. Gradio 4.44.1 with PEFT adapter + 4-bit bitsandbytes. Patched `gradio_client` for Python 3.13 compat. Custom CSS, example prompts, temperature/max-tokens controls. eulogik branding in header and footer.

**Quota strategy:** HF Spaces free tier allows ~2-3 active spaces per namespace. Spread across `eulogik`, `GautamKishore`, and optionally new orgs. Model repos have no quota limit.

### Quality Benchmark (76K iters)

| Prompt | Quality | Notes |
|--------|---------|-------|
| "Kal interview hai, nervous ho raha hoon" | ✅ Good | "Chill maaro, you'll be alright" |
| "Biryani kaise banate hain?" | ✅ Good | Structured recipe with ingredients |
| "Delhi me rehne ke liye kya karna padega?" | ✅ Good | Specific tips on parking, traffic |
| "Weekend pe kya karein?" | ✅ Good | 10 ideas with examples |
| "Mera phone charge nahi ho raha" | ⚠️ OK | Repetitive but actionable |
| "Mujhe Hindi seekhni hai" | ✅ Good | Listening tips, practical advice |
| "Chai peetey hain?" | ✅ Good | Full recipe with instructions |
| "Job nahi mil rahi" | ❌ Poor | Confused, repeating itself |
| "Aap kaise ho?" | ❌ Poor | Switched to full Devanagari |
| "Yaar, aaj ka match dekh liya?" | ⚠️ OK | Understands but Telugu leakage |

### Why We Pivoted from SmolLM2-360M

SmolLM2 hit a hard ceiling after 50K LoRA iterations:
- Val loss plateaued at 1.32 (vs Qwen2.5's 0.781)
- English-only tokenizer: Hindi words tokenized as 3-5 subword tokens
- LoRA modifies only 0.3% (1M/361M) params — fundamentally limited

Qwen2.5-1.5B advantages:
- Already has multilingual representations for 29 languages including Hindi
- 1.5B params means LoRA has more to work with
- 4x larger dataset (376K vs 177K) with diverse domains
- Best val loss improved from 1.168 → 0.781 (33% better)

### Training Data (qwen_v2)

| Source | Samples | Type |
|--------|---------|------|
| ankitdhiman/hinglish-conversations | 201,633 | Natural Hinglish dialogue |
| maya-research/IndicVault (Hindi) | 74,053 | Hindi QA (20 topics) |
| maya-research/IndicVault (Hinglish) | 77,210 | Hinglish QA (20 topics) |
| Sujalvc/hinglish-instruct-dataset | 10,378 | Instruction tuning |
| Subh24ai/yojana-sahayak-instruct | 6,828 | Govt scheme data |
| VishalMysore/cookGPT | 5,938 | Indian recipes |
| **Total** | **376,040** | Deduplicated, messages format |

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
├── .env                        # Credentials (gitignored)
├── .gitignore                  # Git exclusions
├── living.md                   # This document
├── TRAINING_STATUS.md          # Training monitoring guide
├── IMPLEMENTATION_PLAN.md      # 32-week roadmap
├── BRAHMI_Build_GTM_Plan.md    # Original vision doc
├── Strategic_Positioning.md    # Market analysis
├── README.md                   # Project overview
├── models/
│   ├── bharat-tiny-llm-qwen-q4/       # Final Q4 model (828 MB)
│   └── adapters/
│       ├── qwen_v1/                   # v1 adapter (20K iters, val loss 1.168)
│       └── qwen_v2/                   # v2 adapter (76K iters, val loss 0.781)
│           ├── 0075000_adapters.safetensors  # Best checkpoint
│           ├── 0070000_adapters.safetensors  # Previous checkpoint
│           └── ...                         # All 5K interval checkpoints
├── data/
│   ├── processed/
│   │   ├── train.jsonl           # 357K training conversations
│   │   └── valid.jsonl           # 19K validation conversations
│   └── synthetic/
│       └── hinglish_conversations.jsonl  # Archived original data
├── src/
│   └── train.py                  # Training orchestrator
├── scripts/
│   ├── train_bulletproof.py      # Auto-restart training wrapper
│   ├── mass_generate.py          # Parallel data generator
│   ├── generate_final.py         # Sequential generator
│   └── augment_data.py           # Data augmentation
└── configs/
    └── training.yaml             # Training hyperparameters
```

---

## Training History

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
| qwen_v1 | 20,000 | 177K | 1.168 | First pivot, old data |
| qwen_v2 | 50,000 | 376K | 0.835 | New data, batch 2, seq 1024 |
| qwen_v2 | 76,420 | 376K | **0.781** | Final, batch 4, seq 512 |
| qwen_v2 cont. | — | 376K | — | Paused at 76K, resume with --iters 33600 |

### Speed Optimization History

| Config | It/s | Mem GB | Notes |
|--------|------|--------|-------|
| batch=2, seq=1024 | 0.30 | 10.7 | Initial stable config |
| batch=8, seq=256 | 0.24 | 10.0 | Slower due to overhead |
| batch=6, seq=512 | OOM | — | Metal GPU limit |
| batch=4, seq=1024 | OOM | — | Long sequences |
| batch=4, seq=512 | 0.30 | 10.3 | **Final config** |
| batch=4, seq=256 | 0.55 | 6.8 | Fastest but short context |

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
| Week 1 | **Pivot to Qwen2.5-1.5B** | SmolLM2 hit ceiling at val loss 1.32 |
| Week 1 | Use base model (not instruct) | Avoid conflicting RLHF/DPO constraints |
| Week 1 | 16 LoRA layers on Qwen2.5 | More capacity than SmolLM2's 8 |
| Week 2 | **Replaced data pipeline** | 5 curated datasets (376K) vs 177K old |
| Week 2 | max-seq-length=512 | batch=4 OOM at longer sequences |
| Week 2 | Open-core strategy | Apache 2.0 base, commercial enterprise layers |
| Week 2 | **HuggingFace release** (July 6) | 3 repos under eulogik org: model (Q4 MLX), adapter (PEFT), space (Gradio) |
| Week 2 | Model card: transparent about Qwen2.5 base | Brand-forward "Bharat-Tiny-LLM", Qwen in collapsible section |
| Week 2 | Space uses Dockerfile with pre-cached model | Avoids HF Space startup timeout for 3GB model download |
| Week 2 | Patched gradio_client for Python 3.13 | `const in schema` bool bug in gradio-client 1.3.0 |
| Week 2 | Space quota workaround | Spread active spaces across multiple namespaces (eulogik, GautamKishore) |

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| Metal GPU OOM at batch >4 | ⚠️ Known | M4 16GB limit for Qwen2.5-1.5B |
| Job/dating queries produce confusion | ⚠️ Quality | Needs more diverse data |
| Occasional Telugu leakage | ⚠️ Quality | IndicVault Hindi data has Telugu examples |
| "Aap kaise ho?" answered in Devanagari | ⚠️ Quality | Needs more Romanized Hinglish examples |
| OpenRouter free tier rate limited | ⚠️ Known | Cannot generate more synthetic data |
| 76K iters not fully converged | ⚠️ Opportunity | Resume later with lower LR (2e-5) |
| HF Space quota (2-3 active per namespace) | ⚠️ Known | Spread across eulogik, GautamKishore, or new orgs |

---

## Next Actions

### Short term
1. **Evaluate on edge** — Benchmark on Raspberry Pi 5 and Android
2. **Test with low LR** — Resume training from 76K to 110K with 2e-5 LR for convergence
3. **Fix quality gaps** — Add diverse data for jobs, relationships, casual chit-chat
4. ~~**HuggingFace release**~~ ✅ Done — model, adapter, demo space under `eulogik` org
5. ~~**Demo**~~ ✅ Done — Gradio space live at https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM

### Medium term
1. **Add 8 Indic languages** — Tamil, Telugu, Bengali, Marathi, etc. (Qwen2.5 already knows them)
2. **DPO/RLHF** — Human preference tuning for quality leap
3. **BharatTiny-Bench** — Evaluation suite for Indic edge models
4. **BNC-lite** — Syllable-aware tokenizer replacing BPE (pragmatic approach to original vision)
5. **DLR** — Language routing for multi-Indic-language support

### Long term
1. **True BNC** — Vision Transformer tokenizer for Indic scripts (requires custom Metal kernels)
2. **CKH** — Cultural Knowledge Hypernetwork for context-aware generation
3. **EFP** — Edge Federated Personalization
4. **BRAHMI Studio** — No-code fine-tuning platform

---

## Resume Training (Later)

To resume from 76K and train to 110K:

```bash
source venv/bin/activate
export PYTHONUNBUFFERED=1
nohup python3 -u -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-bf16 \
  --train --data data/processed \
  --adapter-path ./models/adapters/qwen_v2 \
  --resume-adapter-file ./models/adapters/qwen_v2/adapters.safetensors \
  --iters 33600 \
  --batch-size 4 --learning-rate 2e-5 \
  --steps-per-eval 500 --save-every 5000 \
  --num-layers 16 --max-seq-length 512 \
  --mask-prompt --seed 42 --val-batches 8 \
  > training_v2.log 2>&1 &
```

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| Week 0 | Initial setup, SmolLM2-360M downloaded | Eulogik |
| Week 0 | Custom tokenizer trained (64K, Unigram) | Eulogik |
| Week 0 | poc_v1–poc_v3 trained | Eulogik |
| Week 0 | prod_v1–prod_v2 trained | Eulogik |
| Week 1 | prod_v3–prod_v4 trained, SmolLM2 plateaued | Eulogik |
| Week 1 | **Pivot to Qwen2.5-1.5B** | Eulogik |
| Week 1 | qwen_v1 trained (20K iters, val loss 1.168) | Eulogik |
| Week 1 | Q4 quantized (828 MB, 20 tok/s) | Eulogik |
| Week 2 | **Replaced data pipeline** — 5 datasets (376K) | Eulogik |
| Week 2 | qwen_v2 trained (50K iters, val loss 0.835) | Eulogik |
| Week 2 | qwen_v2 trained to 76K (val loss 0.781) | Eulogik |
| Week 2 | Q4 fused model (828 MB, **57 tok/s**) | Eulogik |
| Week 2 | living.md updated with v6 results | Eulogik |
| July 6 | HuggingFace release (3 repos under eulogik) | Eulogik |
| July 6 | Gradio Space live + patched for Python 3.13 | Eulogik |
| July 6 | living.md updated with HF release + quota strategy | Eulogik |

---

*This document is updated after every significant milestone. Keep it current.*
