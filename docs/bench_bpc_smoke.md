# Bits-per-character benchmark

- Corpus: `data/processed/valid_gold.jsonl`
- Windows: 60 x 1024 chars (58,210 chars)
- Protocol: one forward pass per window, correct causal alignment, bits/char + bits/byte

| Model | tok/1000 chars | bpc | 95% CI | bits/byte | PPL |
|---|---:|---:|---|---:|---:|
| `Qwen/Qwen3-1.7B-Base` | 600.94 | 1.1973 | 1.1508–1.24 | 0.6891 | 3.979 |
| `/Volumes/KIOXIA 1TB/brahmi-v3-merged` | 492.46 | 1.7094 | 1.6281–1.7914 | 0.9838 | 11.089 |
