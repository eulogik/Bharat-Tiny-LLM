# Data provenance

This repository mixes original benchmark/safety data with research artifacts.
Nothing here is personal data; the v4.0 release does not collect user data.

## v4.0 release artifacts (original, eulogik)

- `data/v4/benchmark_release/` — BharatBench-Voice prompts, base/v4 runs,
  gate report and editorial review. **CC-BY-4.0.**
- `data/v4/cards/cards.jsonl` — 131 vetted bilingual answer cards.
  **CC-BY-4.0** (code Apache-2.0).
- `data/v4/kb/kb.jsonl` — source-linked knowledge-base chunks. **CC-BY-4.0.**
- `data/v4/sft_curated/` — 690-row safety/style SFT set. Built from the
  `eulogik/bharat-v3-chat-sft-v3` org dataset plus curated templates. Research
  artifact only; not used by the shipped v4.0 card/protocol system.
- `data/v4/words/`, `data/v4/h1000/`, `data/v4/hinglish_tokens.json` —
  tokenizer mining artifacts. Derived from public Hinglish/Hindi text and the
  Qwen tokenizer; research only.

## Tokenizer files

`data/bharat_tokenizer_64k.*` is derived from the Qwen tokenizer
(Apache-2.0) plus mined public text. Apache-2.0.

## Synthetic conversations

`data/synthetic/hinglish_conversations*.jsonl` are generated/augmented
assistant conversations created by eulogik for research. No real user data.

## Legacy v2/v3

Legacy training data was derived from public research datasets including
GSM8K-Hindi and IndicVault. Check the upstream dataset licenses before
redistributing those files. The v4.0 system does not depend on them.

## No PII / no secrets

The repository is scanned for phone numbers, emails, Aadhaar-like IDs and
provider tokens before public release. The git history was rewritten on
2026-11-12 to remove two expired credentials found in old notebook commits.
Report anything suspicious via `SECURITY.md`.
