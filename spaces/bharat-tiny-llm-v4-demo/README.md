---
title: Bharat-Tiny-LLM v4.0 — Vetted Answers
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: static
pinned: true
license: apache-2.0
tags:
  - hindi
  - hinglish
  - devanagari
  - edge-ai
  - safety
  - india
  - offline
  - answer-cards
---

# Bharat-Tiny-LLM v4.0 — vetted answers, no model call

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Bharat--Tiny--LLM--v4-ff6f00)](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v4)
[![Dataset](https://img.shields.io/badge/%F0%9F%93%8A%20Benchmark-BharatBench--Voice-2f6b4f)](https://huggingface.co/datasets/eulogik/BharatBench-Voice)
[![GitHub](https://img.shields.io/badge/GitHub-Code-171512)](https://github.com/eulogik/Bharat-Tiny-LLM)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/eulogik/Bharat-Tiny-LLM/blob/main/LICENSE)

This Space demonstrates the layer that makes Bharat-Tiny-LLM v4.0 shippable:
**131 vetted answer cards and 13 fixed health protocols**. Type a question and
the Space returns the curated answer verbatim, in Hinglish or Hindi. There is
no model, no API key and no network call after the page loads.

## Why

A 2B model is fluent but unsafe on facts. In testing it:

- called **CSC** a “Computer Science and Technology” center;
- invented Atal Pension returns and Ayushman coverage;
- recommended Paracetamol for tomato leaves and “doodh ya ghee” for burns;
- told a snake-bite victim to band the wound;
- told a ration-card user to call **108**.

The full system removes model generation from health entirely and answers
common factual questions from vetted cards. The 2B model is only used for the
long tail, with a “verify” note.

## What the full system scores

BharatBench-Voice, 200 Hinglish/Hindi prompts:

| Metric | Base Qwen3.5-2B | v4.0 |
|---|---:|---:|
| Language match | 90.0% | **100%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | **100%** |
| Model calls | 200 | **0** |

Editorial review of 60 non-health rows: **7.68/8 task success, 4.22/5
naturalness, 1.00 safety**. Independent human and clinician review are pending.

## Demo scope

- Only the vetted layer is included. Out-of-set questions show a clear “no
  match” message; the full app would call the 2B model and add a verify note.
- The health protocols are first-aid guidance, **not a diagnosis**. Call 108
  for emergencies.
- The card data is `cards.jsonl`; the protocols are `health_protocols.json`.
- Model: [eulogik/Bharat-Tiny-LLM-v4](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v4)
- Benchmark: [eulogik/BharatBench-Voice](https://huggingface.co/datasets/eulogik/BharatBench-Voice)
- Source: [github.com/eulogik/Bharat-Tiny-LLM](https://github.com/eulogik/Bharat-Tiny-LLM)
  · [gate report](https://github.com/eulogik/Bharat-Tiny-LLM/blob/main/eval/bharatbench_voice/runs/v4_gate_final.md)
- Apache-2.0.
