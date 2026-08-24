#!/bin/bash
# MLX LoRA fine-tuning for Bharat-Tiny-LLM v2
# Step 1: Convert trained PyTorch model → Q4 MLX
# Step 2: Train LoRA adapter
# Step 3: Upload to HF
# Run: source venv/bin/activate && bash scripts/mlx_lora_train.sh

set -e

# Config
HF_PYTORCH="eulogik/Bharat-Tiny-LLM-v2"
MLX_DIR="/Volumes/KIOXIA 1TB/brahmi-mlx-trained"
DATA_DIR="/Volumes/KIOXIA 1TB/mlx_lora_data"
ADAPTER_DIR="/Volumes/KIOXIA 1TB/mlx_lora_adapter"
HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

echo "=== Step 1: Convert trained PyTorch → Q4 MLX ==="
if [ -d "$MLX_DIR" ] && [ -f "$MLX_DIR/config.json" ]; then
    echo "MLX model already exists at $MLX_DIR"
else
    python3 -m mlx_lm convert \
        --hf-path "$HF_PYTORCH" \
        --mlx-path "$MLX_DIR" \
        --quantize \
        --q-group-size 64 \
        --q-bits 4 \
        --q-mode affine
    echo "Converted to $MLX_DIR"
fi

echo ""
echo "=== Step 2: Prepare training data ==="
mkdir -p "$DATA_DIR"

python3 << 'PYEOF'
import json, random, os

with open('data/processed/train_gold_v3.jsonl') as f:
    lines = [l.strip() for l in f if l.strip()]

random.shuffle(lines)
lines = lines[:10000]
split = int(len(lines) * 0.95)

def extract(line):
    try:
        d = json.loads(line)
        if 'messages' in d:
            return ' '.join(m['content'] for m in d['messages'])
        return d.get('text', '')
    except:
        return line

data_dir = '/Volumes/KIOXIA 1TB/mlx_lora_data'
with open(f'{data_dir}/train.jsonl', 'w') as f:
    for l in lines[:split]:
        f.write(json.dumps({'text': extract(l)}) + '\n')
with open(f'{data_dir}/valid.jsonl', 'w') as f:
    for l in lines[split:]:
        f.write(json.dumps({'text': extract(l)}) + '\n')

tr = sum(1 for _ in open(f'{data_dir}/train.jsonl'))
va = sum(1 for _ in open(f'{data_dir}/valid.jsonl'))
print(f'Train: {tr}, Valid: {va}')
PYEOF

echo ""
echo "=== Step 3: LoRA Training ==="
echo "Model: $MLX_DIR"
echo "Data:  $DATA_DIR"
echo ""

export HF_TOKEN
python3 -m mlx_lm lora \
    --model "$MLX_DIR" \
    --train \
    --data "$DATA_DIR" \
    --fine-tune-type lora \
    --num-layers 16 \
    --batch-size 4 \
    --iters 500 \
    --learning-rate 1e-4 \
    --steps-per-report 25 \
    --steps-per-eval 100 \
    --val-batches 20 \
    --save-every 100 \
    --adapter-path "$ADAPTER_DIR" \
    --grad-checkpoint \
    --max-seq-length 512

echo ""
echo "=== Step 4: Upload to HF ==="
python3 << PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get('HF_TOKEN', ''))

# Upload the trained MLX model (with trained embeddings but no LoRA)
api.upload_folder(
    folder_path='$MLX_DIR',
    repo_id='eulogik/Bharat-Tiny-LLM-v2-MLX',
    repo_type='model',
    commit_message='Trained embeddings (from continued pre-training) + Q4 conversion',
)
print('MLX Q4 model uploaded.')

# Upload LoRA adapter
api.upload_folder(
    folder_path='$ADAPTER_DIR',
    repo_id='eulogik/Bharat-Tiny-LLM-v2-MLX',
    repo_type='model',
    path_in_repo='lora_adapter',
    commit_message='LoRA adapter: 500 steps, rank=16, lr=1e-4',
)
print('LoRA adapter uploaded.')
PYEOF

echo ""
echo "=== DONE ==="
echo ""
echo "Test:"
echo "  python3 -m mlx_lm generate --model $MLX_DIR --prompt 'भारत की राजधानी'"
echo "  python3 -m mlx_lm generate --model $MLX_DIR --adapter-path $ADAPTER_DIR --prompt 'भारत की राजधानी'"
