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

- 🇮🇳 Devanagari token injection for 36.8% Hindi compression
- 📊 56% on GSM8K-Hindi (grade-school math in Hindi)
- ⚡ 26.6% fewer bits/char than base Qwen3-1.7B on held-out Hindi

Model: [eulogik/Bharat-Tiny-LLM-v3](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3)
(Apache 2.0) · Offline GGUFs in the same repo.
