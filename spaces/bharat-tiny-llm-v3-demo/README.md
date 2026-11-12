---
title: Bharat-Tiny-LLM v3
emoji: 🗣️
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
tags:
  - hindi
  - hinglish
  - devanagari
  - edge-llm
  - indian-languages
  - conversational
  - inference-providers
---
![Bharat-Tiny-LLM v3 banner](banner.png)

# Bharat-Tiny-LLM v3 — Chat Demo

A 1.7B-parameter bilingual (Hindi / Hinglish / English) LLM. This page is
fully static — it calls the model through Hugging Face Inference Providers
(Featherless AI) straight from your browser. No download, no WebGPU needed.

**Setup:** paste a free read-only HF token
([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens));
it stays in your browser's localStorage.

- 🇮🇳 Devanagari token injection for 36.9% Hindi compression
- 📊 Short Hindi math answers (GSM8K-Hindi is stack-sensitive: 2–35% pinned; see paper)

Model: [eulogik/Bharat-Tiny-LLM-v3](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3)
(Apache 2.0) · Offline GGUFs in the same repo.

> **⚠️ Correction (2026-11-11).** The earlier "26.6% fewer bits/char" claim came
> from a label-shift bug in the benchmark; the corrected run shows v3 is worse
> than base (1.71 vs 1.20 bpc). Do not repeat that number. See `docs/claim_audit.md`.
