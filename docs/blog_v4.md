# We built an offline assistant that refuses to answer

*How a 2B model failed, and what we shipped instead.*

A snake bites a farmer in a village with no clinic. He opens an offline AI
assistant on his phone and asks what to do. The model tells him to band the
wound tightly and wait.

That is not a hypothetical. It is one of the failures we measured while building
Bharat-Tiny-LLM. The same model told a burn victim to apply “doodh ya ghee,”
called CSC a “Computer Science and Technology” center, and told someone to call
**108** to change a name on a ration card.

We could have written a better prompt. We tried. We could have fine-tuned the
model. We tried that too. Neither fixed the problem, because the problem is
structural: a 2B model is fluent enough to sound right and small enough to be
wrong in dangerous ways.

So we changed the product.

## What we shipped

**Bharat-Tiny-LLM v4.0** is a 1.2 GB, Apache-2.0, offline assistant for
Hinglish, Hindi and English. The weights are stock Qwen3.5-2B. The contribution
is the layer around it:

- **13 fixed health protocols** for snake bite, animal bite, chest pain,
  stroke, severe bleeding, breathing difficulty, poisoning, burns, fracture,
  heat stroke, seizure, child fever and diarrhoea. Every protocol includes 108
  and a “do not” list.
- **131 vetted answer cards** for the questions people actually ask: PM-KISAN,
  Fasal Bima, Ayushman, Atal Pension, Sukanya, MGNREGA, ration and labour
  cards, CSC, crop pests, storage, irrigation, recipes, UPI, gas, electricity,
  percentages, multiplication, and more.
- **A domain router** that keeps health, crops, schemes, education and daily
  life apart.
- **Safety guards** that remove human-medicine advice from crop answers, remove
  108 from non-health answers, and retry looping text.
- **The 2B model** only for the long tail, with a verify note.

Health and matched cards never call the model. The answer is fixed text. There
is nothing to sample, hallucinate or truncate.

## The numbers

We built BharatBench-Voice: 200 Hinglish/Hindi prompts across agriculture,
schemes, education, health and daily life. We ran the base model and the v4.0
system on the same prompts, non-streaming, temperature 0.

| Metric | Base Qwen3.5-2B | v4.0 |
|---|---:|---:|
| Language match | 90.0% | **100%** |
| Safety pass | 91.5% | **100%** |
| Live-data honesty | 94.4% | **100%** |
| Math pass | 100% | **100%** |
| Model calls | 200 | **0** |

On the 40 health prompts, safety went from **72.5% to 100%** and language match
from **80% to 100%**. Critical-intent routing scored **62/63 (98.4%)** on the
first run of an unseen paraphrase set; after one fix the regression suite is
63/63.

A 60-row, model-blind editorial review scored the final system **7.68/8 task
success, 4.22/5 naturalness and 1.00 safety**. That review is editorial, not an
independent human study. We are publishing the sheet and looking for
independent reviewers and clinicians.

## Why the benchmark is not the whole story

The 160 non-health benchmark prompts are covered by cards. That makes the
benchmark a **coverage and reliability test for the curated layer**, not an
open-ended model-quality benchmark. We say this in the model card, the dataset
and the release notes.

The honest claim is narrower than “a better 2B model”:

> For the questions that matter most, v4.0 returns a vetted answer instead of a
> generated one. The model handles the rest, and the system says when it is not
> sure.

That is a product decision, not a leaderboard trick.

## What failed

We are keeping the negative results public:

- **Token injection.** Adding 1,000 Hinglish/Devanagari whole words saved
  16.57% tokens on Hinglish, but the 16 GB M4 could not calibrate the new output
  rows alongside LoRA. Generation garbled words like “kaiseain” and
  “documentsyeahi.”
- **SFT.** The stock-base SFT applied the crop template to snake bites and
  child diarrhoea. Train loss looked fine; the product was worse.
- **RAG as a default.** BM25 retrieved a tomato advisory for “CSC centre” and
  the wrong crop for wheat. It is now opt-in `--rag`, not a claimed feature.
- **Free-form voice input.** The best local PolyWhisper Hindi WER was 42.7% on
  FLEURS. Speech output ships; speech input does not.
- **v3 claims.** The 56% GSM8K-Hindi, 26.6% bits/char and 33.8% compression
  numbers were retracted after a label-shift bug and a pinned-protocol rerun.
  v3 is legacy.

## Try it

The vetted layer runs in your browser, with no API and no model:

https://huggingface.co/spaces/eulogik/Bharat-Tiny-LLM-v4-demo

The full system:

```bash
llama-server -m Qwen3.5-2B-Q4_K_M.gguf -c 4096 -np 1 --host 127.0.0.1 --port 8096 &
venv/bin/python scripts/rag_ask.py \
  --server http://127.0.0.1:8096/v1 \
  --model Qwen3.5-2B-Q4_K_M.gguf \
  --query "PM Kisan ka status kaise check karein?"
```

Benchmark, base run, cards and protocols:
https://huggingface.co/datasets/eulogik/BharatBench-Voice

System card:
https://huggingface.co/eulogik/Bharat-Tiny-LLM-v4

Code and docs:
https://github.com/eulogik/Bharat-Tiny-LLM

## What we are looking for

- **Clinicians** to review the 13 health protocols line by line.
- **KVK officers and CSC operators** to review the crop and scheme cards.
- **Independent raters** to re-score the 60-row packet.
- **Real user logs** to expand the cards beyond the benchmark intents.

A small model can be useful. It just should not be trusted with everything.
Knowing when not to answer is the product.

*Apache-2.0. Built by [eulogik](https://eulogik.com).*
