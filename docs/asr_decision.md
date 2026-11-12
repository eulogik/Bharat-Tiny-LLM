# ASR decision — what "voice" can honestly mean for v4

**Date:** 2026-11-12
**Question:** should v4 ship a free-form Hinglish voice assistant built on
PolyWhisper?

## What PolyWhisper actually measures

PolyWhisper (`eulogik/polywhisper`) is Whisper-small + per-language LoRA,
trained on IndicVoices/FLEURS. The public eval files report:

| Model | Hindi WER (FLEURS test) |
|---|---:|
| Whisper-small zero-shot | 69.84% |
| PolyWhisper hi v5 | **42.66%** |
| PolyWhisper hi best-prod | 46.32% |
| PolyWhisper hi beam-5 | 44.98% |
| PolyWhisper hi v9 | 55.10% |

Tamil is worse (80.3% WER in the public eval), Telugu/Bengali/Marathi are
above 100% WER in the small-model baselines. The model is a legitimate
research artifact, but a 43% word error rate is **not** a usable free-form
voice interface. A user saying "mere tamatar ke patte peele ho rahe hain"
will lose roughly two of every five words.

## Why this matters

The v4 plan listed PolyWhisper ASR as a pillar. If we ship it as "voice
input", the first real user will hit transcription garbage, the 2B model will
answer the wrong question, and the launch will produce exactly the kind of
retraction we just spent a week cleaning up. The corrected-benchmark work is
only credible if the next claim is held to the same standard.

## Options

| Option | Hindi WER (expected) | Size / latency | Verdict |
|---|---:|---|---|
| PolyWhisper small + LoRA | 43% measured | ~1 GB, fast | too weak for free text |
| Whisper-small zero-shot | 70% measured | ~1 GB | worse |
| Whisper-medium / large-v3-turbo | not measured locally; typically much lower | 1.5–3 GB, slower | measure before claiming |
| Constrained command grammar | intent accuracy, not WER | small | **ship this first** |
| Text input + TTS output | n/a | tiny | **ship this first** |

## Decision

1. **Do not claim a free-form PolyWhisper voice assistant in v4.0.**
2. Ship **text input + on-device TTS output** as the first voice feature. This
   is genuinely useful for low-literacy users who can listen but not type, and
   it has no ASR risk.
3. For voice *input*, start with a **closed-vocabulary command grammar** over
   the five domains: crop names, scheme names, "mandi rate", "weather",
   "doctor", "108", numbers. Measure **intent accuracy**, not WER. A 43% WER
   can still route "tamatar ... peele" to the crop-health intent if the
   grammar is constrained.
4. Before any free-form voice claim, run the same BharatBench-Voice prompts
   through `whisper-large-v3-turbo` (q5/q8 in whisper.cpp) on the recorded
   audio and report WER + end-to-end task success. Target: **WER ≤ 25%** and
   **task success within 5 points of text input**. If it misses, keep voice
   input as a lab demo, not the headline.
5. The ₹8,000-phone story survives: the text + TTS assistant is small, offline
   and useful; the ASR upgrade is a later, measurable milestone.

## What to build now

- `scripts/voice_loop.py`: push-to-talk → ASR (pluggable backend) → llama.cpp
  → TTS (`say -v Lekha` on macOS; Piper on Linux/Android later), with a
  `--backend` flag so PolyWhisper, Whisper-small and Whisper-large can be
  compared on the same audio.
- `eval/bharatbench_voice/record_audio.py`: record the 200 prompts with the
  speaker sheet, save 16 kHz mono WAV, and keep the transcript.
- A scoring sheet that reports **intent accuracy** for the constrained grammar
  and **WER** for free-form ASR separately. Never collapse them into one
  "voice accuracy" number.

## Honest one-liner for the model card

> v4 ships offline text chat and on-device speech output. Free-form speech
> input is an experimental add-on; the current Hindi ASR is not accurate
> enough for unsupervised use, and we report its WER rather than hiding it.
