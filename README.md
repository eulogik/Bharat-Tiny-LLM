# Bharat-Tiny-LLM

India's first native edge AI for Indian languages.

## Quick Start

```bash
# 1. Setup environment
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start training
python src/train.py
```

## Project Structure

```
Brahmi/
├── .env                    # Credentials (gitignored)
├── .gitignore              # Git exclusions
├── living.md               # Project walkthrough
├── IMPLEMENTATION_PLAN.md  # 32-week roadmap
├── BRAHMI_Build_GTM_Plan.md # Vision document
├── Strategic_Positioning.md # Market analysis
├── models/                 # Model weights
├── data/                   # Training data
├── src/                    # Source code
├── configs/                # Configuration
├── scripts/                # Shell scripts
└── docs/                   # Documentation
```

## Documentation

- **living.md** - Project walkthrough and status
- **IMPLEMENTATION_PLAN.md** - 32-week implementation roadmap
- **BRAHMI_Build_GTM_Plan.md** - Original vision and architecture
- **Strategic_Positioning.md** - Market analysis and positioning

## Hardware Requirements

- Mac Mini M4 16GB (primary development)
- Raspberry Pi 5 (edge deployment target)
- Android device with Snapdragon 6 Gen 1 (mobile testing)

## Cost

- Total infrastructure cost: ~$750
- No cloud compute required
- All training on Mac Mini M4

## Status

See living.md for current project status and next actions.
