# BharatBench-Voice run summary

- Model: `Qwen3.5-2B-Q4_K_M.gguf`
- Prompts: 200 (errors: 0)
- Language match: 0.895
- Junk-free: 1.0
- Safety pass: 0.93
- Live-data honesty: 0.944
- Math pass: 1.0
- Auto task success (0-8, heuristic): 6.08
- TTFT p50/p95: 0.279 / 0.487 s
- Total p50/p95: 9.102 / 17.474 s

## By domain

| Domain | n | Language | Junk-free | Safety | Auto task |
|---|---:|---:|---:|---:|---:|
| agriculture | 40 | 0.9 | 1.0 | 0.925 | 5.22 |
| schemes | 40 | 0.95 | 1.0 | 1.0 | 6.68 |
| education | 40 | 0.925 | 1.0 | 1.0 | 6.92 |
| health | 40 | 0.8 | 1.0 | 0.725 | 4.73 |
| daily | 40 | 0.9 | 1.0 | 1.0 | 6.87 |

Deterministic checks only. Task success and naturalness must be scored
blind on the CSV before making a launch claim.
