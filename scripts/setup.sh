#!/bin/bash
# Bharat-Tiny-LLM Environment Setup
# Run this script to set up your development environment

set -e

echo "=== Bharat-Tiny-LLM Setup ==="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install MLX (Apple Silicon optimized)
echo "Installing MLX framework..."
pip install mlx mlx-lm

# Install llama.cpp (for quantization)
echo "Installing llama.cpp..."
if ! command -v llama-cli &> /dev/null; then
    echo "llama.cpp not found. Installing via brew..."
    brew install llama.cpp
fi

# Create directories
echo "Creating project directories..."
mkdir -p models/checkpoints
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/synthetic
mkdir -p logs

# Download base model
echo "Downloading SmolLM2-360M base model..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = 'HuggingFaceTB/SmolLM2-360M'
save_path = './models/smolm2-360m-base'

print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f'Saving to {save_path}...')
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print('Done!')
"

# Verify installation
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Start training: python src/train.py"
echo "3. Check living.md for project status"
echo ""
