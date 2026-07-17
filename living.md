# Bharat-Tiny-LLM: Living Document
**Last Updated:** July 17, 2026
**Status:** ✅ Training Complete (110K iters, v8) — Cleaned data, fixed generation config, Q4 edge model uploaded — Next: update Space gen params

---

## Project Overview

**What:** India's first native edge AI for Indian languages
**Why:** Zero tiny Indic models exist. Everyone builds skyscrapers, no one builds bicycles.
**Target:** Model running on ₹8,000 phones
**Status:** ✅ Production model v8 complete — Qwen2.5-1.5B fine-tuned on 436K Hinglish/Devanagari conversations (cleaned), fp16 fused uploaded to HF. Live on HuggingFace under `eulogik` org.

---

## Current Status

### Production Model: Bharat-Tiny-LLM v8 (qwen_v2, 110K iters)

| Metric | Value |
|--------|-------|
| Base model | Qwen2.5-1.5B (multilingual, 29 languages) |
| Training data | **436K** cleaned conversations (gold 343K + NebulaByte 100K + smangrul 1K) |
| Method | LoRA (16 layers, rank 16, alpha 32, scale 2.0, dropout 0.05) |
| Training iters | **110,000** (resumed from 24K → 110K) |
| Best val loss | **0.937** (@ iter 22K, old config) / 0.970 (@ iter 35K, this run) |
| Data cleaning | Removed 7,919 contaminated rows (1.8%) → `train_gold_v3.jsonl` (436K) |
| Trainable params | 10.551M (0.684% of 1.5B) |
| Peak memory | 7.3 GB (batch=4, seq=768) |
| Training speed | ~0.18 it/s (batch=4, seq=768) |
| Fusion method | Direct safetensors save (0.0 roundtrip diff, no `save_pretrained`) |
| License | Apache 2.0 (base weights) |

**CRITICAL GENERATION CONFIG (discovered July 17):**
- The model looked "broken" (garbled Thai/CJK/Romanian tokens, echo loops) but this was a **sampling config bug**, NOT a training/data problem.
- Base Qwen2.5-1.5B at temp=0.8 emits garbled out-of-script tokens. Fixed with: `temperature=0.3, top_p=0.85, repetition_penalty=1.25, no_repeat_ngram_size=3`.
- See `configs/generation_config.json` for the canonical config.
- Space MUST use these params or output is unusable.

### HuggingFace Release (July 17 update)

All assets live under the **eulogik** organization:

| Asset | URL | Format | Visibility | Status |
|-------|-----|--------|------------|--------|
| Q4 MLX Model | https://huggingface.co/eulogik/Bharat-Tiny-LLM | 880 MB, Q4 affine (group 64), Apple MLX | Public | ✅ **updated July 17 (v8, val 0.937)** |
| PEFT Adapter | https://huggingface.co/eulogik/Bharat-Tiny-LLM-adapter | 21 MB, transformers+peft compatible | **Private** | archived |
| Adapter Backups | https://huggingface.co/eulogik/Bharat-Tiny-LLM-adapter-backups | All historical checkpoints | **Private** | archived |
| Fused Model | https://huggingface.co/eulogik/Bharat-Tiny-LLM-fused | 3.3 GB, fp16, transformers | Public | ✅ **updated July 17 (v8, val 0.937)** |
| Gradio Demo | https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM | Live demo with eulogik branding | Public | ⚠️ needs gen params update |

**Positioning:** Transparent about Qwen2.5 base (collapsible "Base model details" section), brand-forward with "Bharat-Tiny-LLM" as the name. Full model card with story, badges, training curve, example outputs, limitations, and roadmap.

**Model card tags for SEO:** hinglish, hindi, indian-languages, edge-ai, mac-mini, lora, small-language-model, on-device-ai, apple-silicon, mlx, apache-2.0, lightweight, edge-deployment, low-resource, conversation, india, bharat

**Demo space:** Uses Dockerfile with fused model pre-cached at build time. Python 3.12 + Gradio 4.44.1 + starlette 0.51.0. No PEFT/bitsandbytes. Loads fused model directly via `AutoModelForCausalLM`. Patched `gradio_client._json_schema_to_python_type` for Python 3.12+ bool schema bug. Custom CSS, example prompts, temperature/max-tokens controls. eulogik branding in header and footer.

**Fix history:**
- Starlette 0.52+ broke Gradio 4.44.1's `TemplateResponse` positional arg call → pinned starlette==0.51.0
- `gradio_client` 1.3.0+ crashes on non-dict boolean schema → monkey-patch `_json_schema_to_python_type` at runtime
- PEFT adapter dead end — MLX-trained LoRA weights don't work with PEFT/Transformers despite byte-exact transpose match (fundamental MLX/PEFT LoRA implementation mismatch). Must fuse manually.
- Manual fusion on MPS works (Hinglish detected) but fusion on CPU (for save_pretrained) produces different matmul results due to float16 precision differences → save/load corrupts model behavior
- **Fusion fix**: Save state_dict DIRECTLY to safetensors (bypass `model.save_pretrained()` which corrupted weights with 0.03-0.17 diffs)
- **`--iters` bug**: MLX parameter is total steps for THIS run (starts from 1), NOT cumulative. Must calculate remaining steps manually.
- **caffeinate**: Prevents Mac Mini idle sleep. Without it training runs at ~27% duty cycle.
- **GPU OOM**: batch=4 OOM'd on eval step (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Lowered to batch=2, seq=768 → stable at 8.75 GB.
- **Frequent crashes**: Training process silently exits every 1-6 hours. Mitigated with `--save-every 1000` (vs 5000) to minimize lost progress.

**Quota strategy:** HF Spaces free tier allows ~2-3 active spaces per namespace. Spread across `eulogik`, `GautamKishore`, and optionally new orgs. Model repos have no quota limit.

### Quality Benchmark (150K iters)

| Prompt | 75K Quality | 150K Quality | Notes |
|--------|-------------|-------------|-------|
| "tum chai peete ho?" | ⚠️ OK | ⚠️ OK | Both produce Hinglish but degenerate after ~15 tok |
| "kal ki meeting mein kya hua?" | ✅ Good | ✅ Good | "Bus, aaram se baithe the" (75K) / "nothing much, just discussing plans" (150K) |
| "kya aap mujhe bata sakte hain ki yeh kaam kaise karein?" | — | ✅ Good | Step-by-step reasoning with bullet points |
| "mujhe apna ghar bahut yaad aa raha hai" | ⚠️ OK | ⚠️ OK | Understands context but repeats |
| "mera dost kal ek naya phone khareeda" | — | ⚠️ OK | Understands but degenerates |
| "aapki shadi kab hai?" | — | ❌ Poor | Special character degeneration |

**Overall**: PPL improved 4.56→4.39 (4% better) but generations still show instability—short coherent bursts followed by special character/script degeneration after ~15-20 tokens. Likely needs cleaner training data rather than more iterations.

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
| qwen_v2 | 76,420 | 376K | **0.781** | Batch 4, seq 512 — paused |
| qwen_v2 (V3) | 98,490+1,630 | 376K | 0.929 | Resume at 98K, batch=4, seq=512 → OOM crash at iter 1630 |
| qwen_v2 (V4) | 100K+10,840 | 376K | 0.888 | batch=2, seq=512, 6.9 GB → crashed |
| qwen_v2 (V5) | 108K→150K | 376K | **0.593** (best) / 0.863 (final) | batch=2, seq=768, 8.75 GB, 0.6 it/s |
| **Total** | **150,000** | **376K** | **0.863** | 41,510 new iters from V5, ~20 hrs runtime |

### Speed Optimization History

| Config | It/s | Tok/s | Mem GB | Notes |
|--------|------|-------|--------|-------|
| batch=2, seq=1024 | 0.30 | — | 10.7 | Initial stable config |
| batch=8, seq=256 | 0.24 | — | 10.0 | Slower due to overhead |
| batch=6, seq=512 | OOM | — | — | Metal GPU limit |
| batch=4, seq=1024 | OOM | — | — | Long sequences |
| batch=4, seq=512 | 0.30 | 573 | 10.3 | OOM'd during eval after 90 min |
| batch=2, seq=512 | 0.70 | 717 | 6.9 | Stable, fast, but 50% truncation |
| batch=2, seq=768 | **0.60** | **266** | **8.7** | **✅ Best config — stable, minimal truncation** |
| batch=4, seq=256 | 0.55 | — | 6.8 | Fastest but too short context |

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
| July 7 | **Resume training from 76K → 150K** | Power failure lost 98K run, resumed from last checkpoint |
| July 7 | batch=2, seq=768 as best config | Faster total throughput (717 tok/s) vs batch=4 (573 tok/s), no OOM |
| July 7 | `--save-every 1000` (was 5000) | Training crashes silently every 1-6 hrs — frequent saves minimize loss |
| July 7 | **Backup repos made private** | Adapter and backup repos set private, fused model stays public |
| July 9 | Training completed at 150K iters | Final val loss 0.863, best val loss 0.593 at ~102K |
| July 9 | Direct safetensors fusion | Bypass `save_pretrained()` — 0.0 roundtrip diff confirmed |
| July 9 | PPL benchmark: 4.39 vs 4.56 | 150K marginally better than 75K, but generation quality still degrades |

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| Metal GPU OOM at batch >4 | ⚠️ Known | M4 16GB limit for Qwen2.5-1.5B. Use batch=2. |
| Training silently crashes every 1-6 hrs | ⚠️ Known | No error logs. `--save-every 1000` mitigates. Possibly Metal GPU timeout? |
| Generation degenerates after ~15-20 tok | ⚠️ Quality | Produces special chars/random scripts. Likely noisy training data. |
| Occasional Telugu leakage | ⚠️ Quality | IndicVault Hindi data has Telugu examples |
| OpenRouter free tier rate limited | ⚠️ Known | Cannot generate more synthetic data |
| HF Space quota (2-3 active per namespace) | ⚠️ Known | Spread across eulogik, GautamKishore, or new orgs |
| MLX → PEFT LoRA mismatch | ✅ Dead end | MLX adapter cannot load with PEFT/Transformers. Must fuse manually. |
| CPU fusion corrupts model | ✅ Fixed | Fuse on MPS (not CPU) — float16 matmul differs between CPU and MPS |
| Direct safetensors fusion (bypass save_pretrained) | ✅ Fixed | 0.0 roundtrip diff confirmed |
| Adapter + backup repos made private | ✅ Done | Only fused model is public (needed for HF Space) |
| Val loss unstable (0.59–1.81 range) | ⚠️ Known | val_batches=4 gives noisy estimate. LR 2e-5 may be high for late-stage. |

---

## Next Actions

### Short term
1. **✅ Training complete (150K iters)** — Fused model uploaded, HF Space rebuilt, PPL 4.39
2. **Fix data quality** — Current data produces noisy training (degeneration after ~15 tok). Filter/clean JSONL or regenerate with better prompts.
3. **Reduce LR / cosine schedule** — Try lower LR (1e-5) with cosine decay for stable late-stage convergence
4. **Try different base model** — Llama-3.2-1B or Gemma-2-2B for better Hinglish code-switching
5. **GGUF quantization for edge** — Quantize fused model to Q4/Q8 for Raspberry Pi/Android deployment

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

## Resume Training

To resume from 150K checkpoint:

```bash
source venv/bin/activate
export PYTHONUNBUFFERED=1
nohup /usr/bin/caffeinate -dis /Users/eulogikdeveloper/Documents/Brahmi/venv/bin/python3 -u -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-bf16 \
  --train --data data/processed \
  --adapter-path ./models/adapters/qwen_v2 \
  --resume-adapter-file ./models/adapters/qwen_v2/adapters.safetensors \
  --iters NNNNN \
  --batch-size 2 --learning-rate 1e-5 \
  --steps-per-eval 500 --save-every 1000 \
  --num-layers 16 --max-seq-length 768 \
  --mask-prompt --seed 42 --val-batches 4 \
  > training_v6.log 2>&1 &
```

**Note**: Replace `NNNNN` with desired total steps for this run. The `--iters` parameter counts from 1 (not cumulative). Use `--save-every 1000` (not 5000) to mitigate silent crashes. Use `batch-size 2` (batch=4 causes GPU OOM). `caffeinate -dis` prevents Mac Mini idle sleep.

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
| July 6 | MLX→PEFT adapter dead end — must fuse manually | Eulogik |
| July 6 | CPU fusion ≠ MPS fusion (float16 matmul precision) | Eulogik |
| July 6 | HF Space stable — responds with partial Hinglish | Eulogik |
| July 6 | living.md: "never responds" fixed, model undertrained | Eulogik |
| July 7 | **Training resumed** — 98K→150K, batch=2, seq=768, 0.6 it/s | Eulogik |
| July 7 | Multiple crashes (OOM, silent exit) — `--save-every 1000` | Eulogik |
| July 7 | GPU OOM on eval at batch=4 → fixed at batch=2 | Eulogik |
| July 7 | Adapter + backup repos made private | Eulogik |
| July 9 | **Training complete at 150K iters** | Eulogik |
| July 9 | Fused model uploaded (0.0 roundtrip diff), HF Space rebuilt | Eulogik |
| July 9 | PPL benchmark: 4.39 (150K) vs 4.56 (75K) — marginal improvement | Eulogik |
| July 9 | living.md updated with v7 results | Eulogik |
| July 17 | **Q4 edge model re-quantized & uploaded** — v8 fused (val 0.937) → affine 4-bit (group 64) 880 MB to `eulogik/Bharat-Tiny-LLM`. `quantize_model` + `config.json` `quantization` key (not `quantization_config`). Verified clean Hinglish/Devanagari output at temp=0.3 | Eulogik |

---

*This document is updated after every significant milestone. Keep it current.*
