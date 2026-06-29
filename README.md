# Bharat-Tiny-LLM

India's first native edge AI for Indian languages — built entirely on a Mac Mini M4.

## Results

| Metric | Value |
|---|---|
| Base model | Qwen2.5-1.5B (multilingual, 29 languages) |
| Training data | 177K Hinglish conversations |
| LoRA layers | 16 (full model depth) |
| Val loss | 1.168 (20K iters, still decreasing) |
| Q4 model size | 828 MB |
| Inference speed | ~20 tok/s on Mac Mini M4 |
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

## Project Structure

```
Brahmi/
├── src/train.py              # Training orchestrator
├── scripts/                  # Data generation & augmentation
├── configs/training.yaml     # Training hyperparameters
├── data/processed/           # Training/validation data (177K conversations)
├── models/
│   ├── bharat-tiny-llm-qwen-q4/  # Q4 quantized model (828 MB)
│   └── adapters/qwen_v1/         # LoRA adapter (separate)
├── living.md                 # Project walkthrough
└── IMPLEMENTATION_PLAN.md    # 32-week roadmap
```

## Hardware

- **Training**: Mac Mini M4 16GB, ~5 hours for 20K LoRA iterations
- **Inference**: Runs on Raspberry Pi 5, Android (Snapdragon 6 Gen 1), ₹8,000 phones
- **Cost**: ~$750 total infrastructure (no cloud compute)

## Documentation

- **living.md** — Project walkthrough and status
- **IMPLEMENTATION_PLAN.md** — 32-week implementation roadmap
- **BRAHMI_Build_GTM_Plan.md** — Original vision and architecture
- **Strategic_Positioning.md** — Market analysis and positioning
