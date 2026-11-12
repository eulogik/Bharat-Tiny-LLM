---
license: cc-by-4.0
language:
- hi
- en
task_categories:
- text-generation
- question-answering
tags:
- hindi
- hinglish
- devanagari
- india
- edge-ai
- safety
- benchmark
pretty_name: BharatBench-Voice
size_categories:
- n<1K
---

# BharatBench-Voice

A 200-prompt usability benchmark for Hinglish/Hindi assistants, with a focus on
rural India: agriculture, government schemes, education, health and daily life.
Each prompt has a language, a domain, a safety level, an intent and rubric
points.

This release includes:

- `prompts.json` — 200 prompts (105 Hinglish, 95 Hindi? see the file; the exact
  split is in the JSON).
- `cards.jsonl` — the 131 vetted answer cards used by the measured system.
- `health_protocols.json` — the 13 fixed health protocols and generic referral.
- `runs/base.jsonl` — stock Qwen3.5-2B Q4_K_M, non-streaming, temperature 0,
  `enable_thinking=false`.
- `runs/system_v4.jsonl` — v4.0 system: router + cards + health protocols +
  guards, RAG off, zero model calls for the 200 prompts.
- `runs/v4_gate_final.md` — base vs v4 gate report.
- `runs/editorial_review.csv` — 60-row editorial review (not an independent
  human study).

## Headline numbers

| Metric | Base Qwen3.5-2B | v4.0 |
|---|---:|---:|
| Language match | 90.0% | **100%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | **100%** |
| Model calls | 200 | **0** |

Editorial review of 60 non-health rows: **7.68/8 task success, 4.22/5
naturalness, 1.00 safety**. Independent human and clinician review are pending.

## Important scope note

The v4.0 system answers all 160 non-health benchmark prompts from vetted cards
and all 40 health prompts from fixed protocols. This dataset therefore measures
the reliability/coverage of the curated layer, **not** open-ended model
reasoning. The base run is included for comparison.

## Licence

CC-BY-4.0 for the benchmark prompts and aggregated results. The card and
protocol text is also released under CC-BY-4.0; the code is Apache-2.0 at
[github.com/eulogik/Bharat-Tiny-LLM](https://github.com/eulogik/Bharat-Tiny-LLM).
