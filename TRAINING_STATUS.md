# Training Status - June 30, 2026

## Current Run
- **PID**: Check with `ps aux | grep mlx_lm`
- **Config**: Qwen2.5-1.5B, batch_size=4, max_seq=512, LR=5e-5, 16 LoRA layers
- **Data**: 376K conversations (Hinglish, IndicVault, cookGPT, yojana)
- **Speed**: ~0.26 it/s, ~190 tok/s
- **Peak memory**: 10.3 GB (64% of 16 GB)
- **Target**: 110K iterations
- **Checkpoint save**: every 5K iters to `models/adapters/qwen_v2/`
- **Val eval**: every 500 iters
- **Log file**: `training_v2.log`

## Estimated Time
- 110K iters / 0.26 it/s = ~4.9 days total
- If started ~7PM June 30, finishes ~July 5

## Monitor Commands
```bash
# Latest progress
tail -20 training_v2.log

# Check if running
ps aux | grep mlx_lm | grep -v grep

# Val losses
grep "Val loss" training_v2.log

# Best model
grep "NEW BEST" training_v2.log

# Peak memory
grep "Peak mem" training_v2.log | tail -3

# Saved checkpoints
ls -lt models/adapters/qwen_v2/

# Stop
kill <PID>
```

## After Training Completes
1. Fuse adapter: `python3 -m mlx_lm fuse --model mlx-community/Qwen2.5-1.5B-bf16 --adapter-path ./models/adapters/qwen_v2 --save-path models/bharat-tiny-llm-merged-qwen`
2. Quantize: `python3 -m mlx_lm convert --model models/bharat-tiny-llm-merged-qwen -q --q-bits 4 --mlx-path models/bharat-tiny-llm-qwen-q4`
3. Benchmark: Load `models/bharat-tiny-llm-qwen-q4` and test Hinglish prompts
4. Push to GitHub

## Notes
- batch_size 6+ OOMs on this data (long sequences from IndicVault)
- max_seq 1024 + batch_size 4 also OOMs
- This config (batch_size 4, max_seq 512) is the fastest stable option
- Memory headroom exists (~6 GB) but Metal GPU can't use it due to batch OOM pattern
