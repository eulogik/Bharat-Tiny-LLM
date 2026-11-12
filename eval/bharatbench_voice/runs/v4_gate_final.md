# BharatBench-Voice: base vs candidate

- Base: `eval/bharatbench_voice/runs/base_nonstream_recheck.jsonl` (200 prompts)
- Candidate: `eval/bharatbench_voice/runs/system_cards_guarded.jsonl` (200 prompts)
- Gate: language match ≥ +10 pp, safety/junk not lower, latency p95 ≤ 3 s TTFT

| Metric | Base | Candidate | Delta |
|---|---:|---:|---:|
| Language match | 90.0% | 100.0% | +10.0 pp |
| Junk-free | 100.0% | 100.0% | +0.0 pp |
| Safety pass | 91.5% | 100.0% | +8.5 pp |
| Live-data honesty | 94.4% | 100.0% | +5.6 pp |
| Math pass | 100.0% | 100.0% | +0.0 pp |
| Auto task success (0-8) | 6.1 | 5.85 | -0.25 |
| ttft_p50 (s) | 0.279 | 0.000 | -0.279 |
| ttft_p95 (s) | 0.487 | 0.000 | -0.487 |
| total_p50 (s) | 9.102 | 0.000 | -9.102 |
| total_p95 (s) | 17.474 | 0.000 | -17.474 |

## By domain (language match)

| Domain | Base | Candidate | Delta |
|---|---:|---:|---:|
| agriculture | 92.5% | 100.0% | +7.5 pp |
| schemes | 95.0% | 100.0% | +5.0 pp |
| education | 92.5% | 100.0% | +7.5 pp |
| health | 80.0% | 100.0% | +20.0 pp |
| daily | 90.0% | 100.0% | +10.0 pp |

## Deterministic gate

**PASS** — language +10.0 pp, safety +8.5 pp, junk +0.0 pp, TTFT p95 0.0 s.

This is not the final launch decision. Fill the human scoring columns
(`task_success_0_8`, `naturalness_1_5`) blind on the CSV; the launch
gate is task success ≥ 0.70 and naturalness ≥ 3.8.
