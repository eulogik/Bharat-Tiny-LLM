# BharatBench-Voice run summary

- Model: `Qwen3.5-2B-Q4_K_M.gguf`
- Prompts: 200 (errors: 0)
- Language match: 0.95
- Junk-free: 1.0
- Loop-free (5-gram < 0.50): 1.0
- Safety pass: 1.0
- Live-data honesty: 1.0
- Math pass: 1.0
- Auto task success (0-8, heuristic): 6.59
- TTFT p50/p95: 0.229 / 0.339 s
- Total p50/p95: 6.426 / 6.877 s

## By domain

| Domain | n | Language | Junk-free | Safety | Auto task |
|---|---:|---:|---:|---:|---:|
| agriculture | 40 | 0.95 | 1.0 | 1.0 | 6.25 |
| schemes | 40 | 0.95 | 1.0 | 1.0 | 6.68 |
| education | 40 | 0.925 | 1.0 | 1.0 | 6.97 |
| health | 40 | 1.0 | 1.0 | 1.0 | 6.2 |
| daily | 40 | 0.925 | 1.0 | 1.0 | 6.87 |

Deterministic checks only. Task success and naturalness must be scored
blind on the CSV before making a launch claim.
