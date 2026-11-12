# BharatBench-Voice run summary

- Model: `Qwen3.5-2B-Q4_K_M.gguf`
- Prompts: 200 (errors: 0)
- Language match: 0.945
- Junk-free: 1.0
- Safety pass: 0.99
- Live-data honesty: 1.0
- Math pass: 1.0
- Auto task success (0-8, heuristic): 6.41
- TTFT p50/p95: 0.229 / 0.339 s
- Total p50/p95: 6.437 / 6.877 s

## By domain

| Domain | n | Language | Junk-free | Safety | Auto task |
|---|---:|---:|---:|---:|---:|
| agriculture | 40 | 0.925 | 1.0 | 0.95 | 5.35 |
| schemes | 40 | 0.95 | 1.0 | 1.0 | 6.68 |
| education | 40 | 0.925 | 1.0 | 1.0 | 6.97 |
| health | 40 | 1.0 | 1.0 | 1.0 | 6.2 |
| daily | 40 | 0.925 | 1.0 | 1.0 | 6.87 |

Deterministic checks only. Task success and naturalness must be scored
blind on the CSV before making a launch claim.
