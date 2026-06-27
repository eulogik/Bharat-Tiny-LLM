# Bharat-Tiny-LLM: Living Document
**Last Updated:** Week 0, Day 0
**Status:** Setup Phase

---

## Project Overview

**What:** India's first native edge AI for Indian languages
**Why:** Zero tiny Indic models exist. Everyone builds skyscrapers, no one builds bicycles.
**Target:** 350M parameter model running on ₹8,000 phones

---

## Current Status

### Week 0: Project Initialization

| Task | Status | Notes |
|------|--------|-------|
| .env credentials created | ✅ Done | Gitignored, secure |
| .gitignore configured | ✅ Done | Protects secrets + artifacts |
| Project structure created | ⏳ Pending | See below |
| Living.md created | ⏳ Pending | This document |
| Git initialized | ⏳ Pending | Private repo |

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
| Week 0 | Custom tokenizer | 3-5× better than Llama for Indic scripts |

---

## Training Progress

### Phase 0: Setup (Week 1-2)

**Status:** Not started

**Tasks:**
- [ ] Install MLX framework
- [ ] Download SmolLM2-360M
- [ ] Train custom tokenizer
- [ ] Create 10K Hinglish samples
- [ ] Fine-tune POC model
- [ ] Record metrics

**Metrics to Track:**
- Training speed (tok/s)
- Peak memory usage (GB)
- Validation loss
- Sample quality (subjective)

---

## Data Inventory

### Downloaded

| Dataset | Size | Status | Source |
|---------|------|--------|--------|
| SmolLM2-360M | 720MB | Pending | HuggingFace |
| AI Kosh samples | - | Pending | Government |
| IndicCorp v2 | - | Pending | AI4Bharat |

### Generated

| Dataset | Size | Status | Cost |
|---------|------|--------|------|
| Hinglish dialogues | - | Pending | ~$28 |
| Cultural Q&A | - | Pending | ~$10 |
| Instruction pairs | - | Pending | ~$20 |

---

## Experiments Log

### Experiment 001: Baseline

| Metric | Value |
|--------|-------|
| Date | - |
| Model | SmolLM2-360M |
| Data | 10K Hinglish |
| Training time | - |
| Peak memory | - |
| Validation loss | - |
| Samples | - |

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| None yet | - | - |

---

## Next Actions

1. **Immediate (Today)**
   - Run `scripts/setup.sh` to install dependencies
   - Download SmolLM2-360M base model
   - Create project directories

2. **This Week**
   - Train custom tokenizer on Indic data
   - Collect 100MB of Indic text samples
   - Create 10K Hinglish training samples

3. **Next Week**
   - Fine-tune POC model
   - Record training metrics
   - Generate sample outputs

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

---

*This document is updated after every significant milestone. Keep it current.*
