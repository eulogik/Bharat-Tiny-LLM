# Bharat-Tiny-LLM: Living Document
**Last Updated:** Week 1, Day 1
**Status:** POC Phase — Model Trained & Quantized

---

## Project Overview

**What:** India's first native edge AI for Indian languages
**Why:** Zero tiny Indic models exist. Everyone builds skyscrapers, no one builds bicycles.
**Target:** 350M parameter model running on ₹8,000 phones
**Status:** ✅ POC complete — Hinglish-speaking model trained on Mac Mini M4 16GB

---

## Current Status

### Week 1: POC Complete 🎉

| Task | Status | Notes |
|------|--------|-------|
| Environment setup | ✅ Done | MLX with Metal GPU, venv |
| SmolLM2-360M base downloaded | ✅ Done | 361.8M params, 693 MB FP16 |
| Custom tokenizer trained | ✅ Done | 64K vocab, Unigram, Indic corpus |
| Hinglish dataset generated | ✅ Done | 1,111 conversations via OpenRouter |
| LoRA fine-tuning (3 rounds) | ✅ Done | prod_v2: 2000 iters from prod_v1 |
| Model merged & exported | ✅ Done | FP16 (698 MB) + Q4 (202 MB) |
| Q4 quantization | ✅ Done | 4.5 bits/weight, 202 MB |
| GitHub repo created | ✅ Done | https://github.com/eulogik/Bharat-Tiny-LLM |
| All code pushed | ✅ Done | Main branch with full pipeline |

### Git Status
- Repository: https://github.com/eulogik/Bharat-Tiny-LLM (private)
- Branch: main
- All scripts, data, configs tracked (models gitignored)

---

## Hardware Setup

**Primary Development Machine:**
- Model: Mac Mini M4
- RAM: 16GB unified memory
- Storage: Check available space
- OS: macOS

**Software Requirements:**
- Python 3.11+
- MLX framework (Apple Silicon optimized)
- llama.cpp (for quantization)
- Git

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
├── models/                 # Model weights (gitignored)
│   ├── smolm2-360m-base/   # Base model
│   └── checkpoints/        # Training checkpoints
├── data/                   # Training data (gitignored)
│   ├── raw/                # Downloaded datasets
│   ├── processed/          # Tokenized data
│   └── synthetic/          # API-generated Hinglish
├── src/                    # Source code
│   ├── train.py            # Training scripts
│   ├── evaluate.py         # Evaluation suite
│   ├── export.py           # Quantization + export
│   └── utils.py            # Helper functions
├── notebooks/              # Jupyter notebooks
│   ├── 01_tokenizer.ipynb  # Tokenizer experiments
│   ├── 02_training.ipynb   # Training visualization
│   └── 03_evaluation.ipynb # Benchmarking
├── configs/                # Configuration files
│   ├── training.yaml       # Training hyperparameters
│   └── model.yaml          # Model architecture
├── scripts/                # Shell scripts
│   ├── setup.sh            # Environment setup
│   ├── train.sh            # Training launcher
│   └── export.sh           # Export pipeline
└── docs/                   # Documentation
    ├── architecture.md     # BRAHMI architecture
    ├── data_strategy.md    # Data collection guide
    └── deployment.md       # Edge deployment
```

---

## Credentials & Access

**Stored in .env (gitignored):**

| Service | Purpose | Access Level |
|---------|---------|--------------|
| OpenRouter | Synthetic data generation | API calls |
| GitHub (Eulogik) | Private repo, code | Full admin |
| GitHub Actions | CI/CD workflows | Write access |
| PyPI | Package publishing | Upload |
| Hugging Face | Model hosting | Push models |

**How to use:**
```bash
# Load credentials in Python
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
hf_token = os.getenv('HF_TOKEN')
```

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Week 0 | Use SmolLM2-360M as base | Apache 2.0, 360M params, strong English |
| Week 0 | Mac Mini M4 only | $0 cloud cost, democratizes development |
| Week 0 | Warm-start training | Cannot train from scratch (4+ years) |
| Week 0 | Custom tokenizer | 64K vocab, Unigram, trained on 1.36GB Indic corpus |
| Week 1 | Switch to messages format w/ mask-prompt | Better QA learning vs text format |
| Week 1 | Use OpenRouter free models | qwen/qwen3-8b for cheap data gen |
| Week 1 | Q4 quantization via MLX quantize | 202 MB model fits on edge devices |

---

## Training Progress

### Round 1: poc_v1 (Text Format)
- Model: SmolLM2-360M + LoRA (8 layers, rank 16)
- Data: 227 conversations (text format)
- Training: 300 iters
- Loss: 3.50 → 1.03
- Memory: 1.5 GB peak
- Result: Model speaks Hinglish but format was wrong

### Round 2: poc_v2 (Text Format, more data)
- Data: 351 conversations
- Training: 500 iters (resumed from poc_v1)
- Loss: 2.19 → 0.39
- Result: Better but still text format

### Round 3: poc_v3 (Messages Format)
- Data: 351 conversations (messages format)
- Training: 800 iters (from scratch)
- Loss: 3.42 → 0.38
- Chat template fixed for prompt masking
- Result: Proper QA learning

### Round 4: prod_v1 (Messages, more data)
- Data: 561 conversations
- Training: 1200 iters (from scratch)
- Loss: 3.45 → 0.39
- Improved coherence

### Round 5: prod_v2 (Final, all data)
- Data: 1,111 conversations
- Training: 2000 iters (resumed from prod_v1)
- Learning rate: 5e-5 → 3e-5
- Loss: 2.34 → 0.81
- Memory: 1.5 GB peak (very stable)
- Speed: ~3-4 iters/sec

---

## Data Inventory

### Downloaded

| Dataset | Size | Status | Source |
|---------|------|--------|--------|
| SmolLM2-360M | 693 MB | ✅ Downloaded | HuggingFace |
| Hindi Wikipedia | 183 MB | ✅ Downloaded | Wikimedia |
| Tamil Wikipedia | 304 MB | ✅ Downloaded | Wikimedia |
| Telugu Wikipedia | 468 MB | ✅ Downloaded | Wikimedia |
| Bengali Wikipedia | 350 MB | ✅ Downloaded | Wikimedia |
| Marathi Wikipedia | 134 MB | ✅ Downloaded | Wikimedia |
| Indic Corpus (merged) | 1.36 GB | ✅ Created | For tokenizer training |

### Generated

| Dataset | Size | Status | Cost |
|---------|------|--------|------|
| Custom tokenizer | 64K vocab | ✅ Trained | Free |
| Hinglish conversations | 1,111 samples | ✅ Generated | ~$0 (free API) |

## Experiments Log

### Final Model: prod_v2 (Bharat-Tiny-LLM v1)

| Metric | Value |
|--------|-------|
| Base model | SmolLM2-360M (361.8M params) |
| Training data | 1,111 Hinglish conversations |
| Method | LoRA (8 layers, rank 16) |
| Training iters | 2000 (resumed from prod_v1) |
| Final loss | 0.81 |
| Peak memory | 1.5 GB |
| Quantized size | 202 MB (Q4) |
| Inference speed | 54 tok/s (Mac Mini M4) |
| Expected Pi 5 speed | ~40-50 tok/s |
| Expected Android speed | ~30-40 tok/s |

### Quality Assessment

| Category | Rating | Notes |
|----------|--------|-------|
| Hinglish recognition | ✅ Good | Understands Hinglish prompts |
| Hinglish response | ⚠️ Fair | Attempts Hinglish, often garbled |
| Answer relevance | ❌ Needs work | Many hallucinated/irrelevant answers |
| Coherence | ❌ Needs work | Limited by small dataset |
| Cultural knowledge | ❌ Not yet | No cultural data injected yet |

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| Small training dataset (1,111) | ⚠️ Known limit | Need 10K+ for production quality |
| Free API data quality | ⚠️ Poor | Use paid APIs or curated data |
| No validation set loading | ⚠️ MLX issue | Training proceeds without validation |
| Chat template "ASSISTANT:" in output | ⚠️ Format issue | Prompt needs proper formatting |
| Tokenizer not integrated | ❌ Not done | Need to replace SmolLM2 tokenizer |

---

## Next Actions

### Short term (Next week)
1. Generate 10K+ high-quality Hinglish conversations (use DeepSeek API, ~$28)
2. Curate and clean the training data (remove poor quality samples)
3. Continue training with larger dataset
4. Integrate custom tokenizer into the model
5. Add cultural knowledge data (festivals, greetings) 
6. Create demo video: "Chat in Hinglish on ₹8,000 phone"

### Medium term (Next month)
1. Implement RLHF/DPO for quality improvement
2. Add 8 more languages (Tamil, Telugu, Bengali, Marathi)
3. Create BharatTiny-Bench evaluation suite
4. Release on HuggingFace
5. Create web demo with Gradio
6. Raspberry Pi 5 deployment image

### Long term (Next quarter)
1. Add CKH (Cultural Knowledge Hypernetwork)
2. Add EFP (Edge Federated Personalization)
3. Multimodal (VLM for documents)
4. BRAHMI Studio - no-code fine-tuning

---

## Notes

### Security
- Never commit .env file
- Never paste credentials in chat
- Use environment variables in code
- Rotate tokens if compromised

### Training Tips
- Run heavy training overnight
- Monitor memory usage (kill if >14GB)
- Save checkpoints every 1000 steps
- Log everything to training.log

### Resources
- MLX docs: https://ml-explore.github.io/mlx/
- llama.cpp: https://github.com/ggerganov/llama.cpp
- SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-360M

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| Week 0, Day 0 | Initial setup | Eulogik |
| Week 0, Day 0 | Created project structure, scripts, config | Eulogik |
| Week 0, Day 0 | Initialized git repository | Eulogik |
| Week 0, Day 0 | SmolLM2-360M downloaded, MLX working | Eulogik |
| Week 0, Day 0 | Custom tokenizer trained (64K, Unigram) | Eulogik |
| Week 0, Day 0 | Model converted to MLX, 109 tok/s baseline | Eulogik |
| Week 0, Day 0 | poc_v1 trained (300 iters, text format) | Eulogik |
| Week 0, Day 0 | poc_v2 trained (500 iters, more data) | Eulogik |
| Week 0, Day 0 | poc_v3 trained (800 iters, messages format) | Eulogik |
| Week 0, Day 0 | prod_v1 trained (1200 iters, 561 convos) | Eulogik |
| Week 1, Day 0 | prod_v2 trained (2000 iters, 1111 convos) | Eulogik |
| Week 1, Day 0 | Q4 quantization (202 MB, 54 tok/s) | Eulogik |
| Week 1, Day 0 | GitHub repo created and code pushed | Eulogik |
| Week 1, Day 1 | POC complete - model speaks Hinglish | Eulogik |

---

## Git Repository Setup

**To create remote repo and push:**

```bash
# 1. Create repo on GitHub (using gh CLI)
gh repo create eulogik/Bharat-Tiny-LLM --private --source=. --remote=origin --push

# Or manually:
git remote add origin git@github.com:eulogik/Bharat-Tiny-LLM.git
git push -u origin main
```

**To use GitHub token:**
```bash
# Set remote with token
git remote set-url origin https://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@github.com/eulogik/Bharat-Tiny-LLM.git
```

---

*This document is updated after every significant milestone. Keep it current.*
