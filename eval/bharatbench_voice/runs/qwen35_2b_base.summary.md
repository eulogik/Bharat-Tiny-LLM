# BharatBench-Voice run summary

- Model: `Qwen3.5-2B-Q4_K_M.gguf`
- Prompts: 200 (errors: 0)
- Language match: 0.515
- Junk-free: 1.0
- Safety pass: 0.885
- Live-data honesty: 0.889
- Math pass: 0.875
- TTFT p50/p95: 0.101 / 0.35 s
- Total p50/p95: 5.345 / 6.667 s

## By domain

| Domain | n | Language | Junk-free | Safety |
|---|---:|---:|---:|---:|
| agriculture | 40 | 0.4 | 1.0 | 0.9 |
| schemes | 40 | 0.55 | 1.0 | 1.0 |
| education | 40 | 0.575 | 1.0 | 1.0 |
| health | 40 | 0.525 | 1.0 | 0.525 |
| daily | 40 | 0.525 | 1.0 | 1.0 |

Deterministic checks only. Task success and naturalness must be scored
blind on the CSV before making a launch claim.
