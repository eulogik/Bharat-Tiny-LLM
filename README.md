# Bharat-Tiny-LLM

India's first native edge AI for Indian languages — built entirely on a Mac Mini M4.

## Results

| Metric | Value |
|---|---|
| Base model | Qwen2.5-1.5B (multilingual, 29 languages) |
| Training data | 376K Hinglish conversations (5 curated datasets) |
| LoRA layers | 16 (full model depth) |
| Training iters | 76,420 |
| **Best val loss** | **0.781** |
| Q4 model size | 828 MB |
| Inference speed | **~57 tok/s** on Mac Mini M4 |
| Training cost | $0 cloud compute |
| License | Apache 2.0 (base weights) |

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate
pip install mlx-lm

# 2. Run inference
python3 -c "
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
model, tokenizer = load('models/bharat-tiny-llm-qwen-q4')
sampler = make_sampler(temp=0.7)
response = generate(model, tokenizer, prompt='USER: Chai peete hain?\nASSISTANT:', max_tokens=80, sampler=sampler)
print(response)
"
```

## Training Data

| Source | Samples | Domain |
|--------|---------|--------|
| Hinglish Conversations | 202K | Natural dialogue |
| IndicVault (Hindi + Hinglish) | 151K | 20 topic QA |
| Hinglish Instruct | 10K | Instructions |
| cookGPT | 6K | Indian recipes |
| Yojana Sahayak | 7K | Government schemes |
| **Total** | **376K** | Mixed |

## Project Structure

```
Brahmi/
├── data/processed/               # 376K training conversations
├── models/
│   ├── bharat-tiny-llm-qwen-q4/  # Q4 quantized model (828 MB, 57 tok/s)
│   └── adapters/
│       ├── qwen_v1/              # v1 (20K iters, val loss 1.168)
│       └── qwen_v2/              # v2 (76K iters, val loss 0.781)
├── scripts/train_bulletproof.py  # Auto-restart training wrapper
├── living.md                     # Full walkthrough
└── IMPLEMENTATION_PLAN.md        # 32-week roadmap
```

## Hardware

- **Training**: Mac Mini M4 16GB, ~3.3 days for 76K LoRA iterations
- **Inference**: Runs on Raspberry Pi 5, Android (Snapdragon 6 Gen 1), ₹8,000 phones
- **Cost**: ~$750 total infrastructure (zero cloud compute)

## Documentation

- **living.md** — Full project walkthrough, decisions, and next steps
- **IMPLEMENTATION_PLAN.md** — 32-week implementation roadmap
- **BRAHMI_Build_GTM_Plan.md** — Original vision and architecture
- **Strategic_Positioning.md** — Market analysis and positioning
