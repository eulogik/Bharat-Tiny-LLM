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
  - quantized
  - gguf
  - webgpu
  - in-browser
  - indian-languages
  - offline
  - transformers.js
---

# Bharat-Tiny-LLM v3 — In-Browser Demo

A 1.7B-parameter bilingual (Hindi / Hinglish / English) edge LLM that runs
**entirely in your browser** via WebGPU. No server, no API key, fully private.

- 🇮🇳 Devanagari token injection for 21.6% Hindi compression
- 📊 56%+ on GSM8K-Hindi (grade-school math in Hindi)
- ⚡ 26.6% fewer bits/char than base Qwen3-1.7B on held-out Hindi
- 🔒 Privacy-first: inference never leaves your device

The first load downloads the ~1 GB Q4 GGUF (then cached). Works best in
Chrome/Edge with WebGPU enabled.

Model: [eulogik/Bharat-Tiny-LLM-v3](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v3)
