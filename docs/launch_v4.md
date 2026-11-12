# v4.0 launch kit

**One-line pitch:** The offline Hinglish assistant that refuses to invent
medicine — or a scheme amount.

**Live demo (no API, no model):** https://huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
**HF system card:** https://huggingface.co/GautamKishore/Bharat-Tiny-LLM-v4
**Benchmark dataset:** https://huggingface.co/datasets/GautamKishore/BharatBench-Voice

**Long pitch:** Bharat-Tiny-LLM v4.0 is a 1.2 GB, Apache-2.0, offline assistant
for rural India. It answers the questions people actually ask from 13 vetted
health protocols and 131 vetted answer cards, then falls back to a 2B model for
the long tail. On a 200-prompt Hinglish/Hindi benchmark it scores 100% language
match, 100% safety, 100% live-data honesty, and 100% math — with **zero model
calls**. Health questions never touch the model.

**Why it matters:** A 2B model is useful, but it will tell a snake-bite victim
to band the wound and call CSC a “Computer Science” center. We measured those
failures, removed model generation where it is dangerous, and shipped a system
that is honest about what it knows.

---

## Verified numbers

| Claim | Number |
|---|---|
| Benchmark | BharatBench-Voice, 200 prompts, 5 domains |
| Language match | 90.0% base → **100%** |
| Safety | 91.5% base → **100%** |
| Live-data honesty | 94.4% base → **100%** |
| Math | 100% → **100%** |
| Model calls on benchmark | 200 → **0** |
| Cards used | **160 / 200** |
| Health safety (40 prompts) | 72.5% → **100%** |
| Health language match | 80% → **100%** |
| Critical-intent routing, first unseen run | **62/63 (98.4%)** |
| Editorial 60-row review | **7.68/8 task, 4.22/5 naturalness, 1.0 safety** |
| Size / license | **1.2 GB Q4_K_M, Apache-2.0** |

**Do not claim:** human-validated, clinically reviewed, RAG-grounded, voice
input, 56% / 26.6% / 33.8%, or “state of the art”. The human and clinician
reviews are pending.

---

## X / Twitter thread

**1/**

A 2B model told a snake-bite victim to band the wound.
It called CSC a “Computer Science” center.
It told someone to call 108 to change a ration-card name.

So we stopped letting it answer those questions.

Bharat-Tiny-LLM v4.0 is live. 1.2 GB. Offline. Apache-2.0. 🧵

**2/**

What we built:

• 13 fixed health protocols (snake bite, chest pain, stroke, bleeding…)
• 131 vetted answer cards (schemes, crops, daily tasks, maths)
• 1 domain router
• 3 safety guards
• a 2B model for the long tail

Health and cards = **zero model calls**.

**3/**

The benchmark (200 Hinglish/Hindi prompts):

• Language match: 90% → 100%
• Safety: 91.5% → 100%
• Live-data honesty: 94.4% → 100%
• Math: 100%
• Model calls: 200 → 0

**4/**

The best part is what it refuses to do:

• No medicine names for crops
• No invented scheme amounts
• No 108 for a ration card
• No crop advice for a snake bite
• No “I’m sure” when it isn’t

**5/**

This is not another fine-tune.

We tried SFT. We tried a new tokenizer. We tried RAG. All failed in public, and
we documented why:
github.com/eulogik/Bharat-Tiny-LLM/blob/main/docs/v4_ab_plan.md

Try the vetted layer (no API): huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo

The product is the safety layer, not a bigger model.

**6/**

Run it:

llama-server -m Qwen3.5-2B-Q4_K_M.gguf
python scripts/rag_ask.py --query “PM Kisan ka status kaise check karein?”

Offline. ₹8,000-phone class. Apache-2.0.

Feedback wanted — especially from clinicians, KVK officers and CSC operators.

---

## LinkedIn post

Most AI demos fail in the same place: they are confident when they should be
quiet.

We built Bharat-Tiny-LLM v4.0 for rural India: 1.2 GB, offline, Apache-2.0,
Hinglish-first. Instead of letting a 2B model improvise, it answers from 13
vetted health protocols and 131 vetted answer cards, then uses the model only
for the long tail.

On a 200-prompt benchmark:
• Language match 90% → 100%
• Safety 91.5% → 100%
• Live-data honesty 94.4% → 100%
• Math 100%
• Model calls 200 → 0

The failures that drove this were concrete: Paracetamol for tomato leaves,
“doodh ya ghee” for burns, CSC expanded as “Computer Science”, a ration-card
answer telling users to call 108.

We tried SFT and a new tokenizer first. Both failed. The negative results are
public. The lesson: for a small model, **knowing when not to answer is the
product**.

Demo: huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
Benchmark: huggingface.co/datasets/GautamKishore/BharatBench-Voice
Repo + safety docs: github.com/eulogik/Bharat-Tiny-LLM

We are looking for clinicians and KVK/CSC reviewers. Independent human and
clinical review are the next gates.

---

## Show HN

**Show HN: Offline Hinglish assistant that refuses to improvise medicine
(1.2 GB, Apache-2.0)**

We built a small offline assistant for Indian users. The model is stock
Qwen3.5-2B; the interesting part is the system around it.

What we found: a 2B model is fluent but unsafe on facts. It recommended
Paracetamol for tomato leaves, “doodh ya ghee” for burns, and told a
snake-bite victim to band the wound. Prompting and RAG did not fix it.

So we removed model generation from health entirely (13 fixed protocols) and
added 131 vetted answer cards for common schemes, crops, daily tasks and maths.
Cards and protocols return verbatim; the model handles the long tail.

BharatBench-Voice: 200 Hinglish/Hindi prompts. Language 90%→100%, safety
91.5%→100%, live-data 94.4%→100%, math 100%, model calls 200→0. Editorial
review of 60 rows: 7.68/8 task, 4.22/5 naturalness.

It runs as a 1.2 GB Q4 GGUF with llama.cpp. No network, no Play Services.
Apache-2.0.

Try the vetted layer in the browser (no API): https://huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
Benchmark: https://huggingface.co/datasets/GautamKishore/BharatBench-Voice

Happy to answer questions about the failure modes, the benchmark, or why we
parked SFT/tokenizer/RAG. Independent human and clinician review are pending;
we are not claiming those.

---

## Blog outline (900–1200 words)

1. **The snake-bite answer.** A 2B model told a victim to band the wound.
   Open with the failure, not the architecture.
2. **We measured it.** 200 prompts, 5 domains, base vs system. Show the table.
3. **Why fine-tuning failed.** Injected tokenizer garbled words; SFT applied the
   crop template to snake bites. Negative results are part of the story.
4. **The fix: don’t generate when you shouldn’t.** 13 protocols + 131 cards +
   router + guards. Explain the card/protocol boundary.
5. **What the benchmark actually measures.** Card coverage, not open-ended
   reasoning. Be explicit about the scope.
6. **What is still weak.** Long-tail model answers, no clinician review, RAG
   retrieval failures, no free-form voice input.
7. **How to run it.** Three commands.
8. **What’s next.** Independent human review, clinician sign-off, real user
   logs, voice output.

---

## Demo script (60 seconds)

```bash
# 1. Health: no model call, fixed protocol
python scripts/rag_ask.py --query "Saanp ne kaata hai, kya karein?"

# 2. Factual card: no model call, correct CSC
python scripts/rag_ask.py --query "CSC center kya kaam karta hai?"

# 3. Hindi card: correct PM-KISAN amount, no invented date
python scripts/rag_ask.py --query "मुझे किसान सम्मान निधि की अगली किस्त कब मिलेगी?"

# 4. Live-data honesty: no invented mandi price
python scripts/rag_ask.py --query "Mandi mein aaj tamatar ka rate kya hai?"
```

Show the `Answer card` / `Safety protocol` line and the `[0.0s]` latency. That is
the story: the right answer, instantly, without the model.

---

## Objection handling

**“This is just an FAQ.”**
For the covered intents, yes — deliberately. Those are the questions where a
wrong answer is expensive. The model still handles everything else, and the
system says when it is unsure.

**“The benchmark is cards covering the test set.”**
Correct. It is a coverage/reliability test, not an open-ended model benchmark.
We say that in the model card and the release summary.

**“Why not a bigger model?”**
A 4B model does not fit the ₹8,000-phone / 1.2 GB constraint. The safety layer
is the cheaper intervention.

**“Is it clinically reviewed?”**
Not yet. That is the next gate. Do not use it as a medical device.

**“Does it work offline?”**
Yes, after the GGUF and repo are on disk. The hosted demo uses an API; do not
call that offline.

---

## Assets checklist

- [x] Static demo with no API/model: https://huggingface.co/spaces/GautamKishore/Bharat-Tiny-LLM-v4-demo
- [x] Screenshot: `assets/demo_v4.png`
- [x] Repo README hero updated with v4 SVG, demo, dataset.
- [x] HF system card: https://huggingface.co/GautamKishore/Bharat-Tiny-LLM-v4
- [x] HF dataset: https://huggingface.co/datasets/GautamKishore/BharatBench-Voice
- [ ] 30-second screen recording: health → CSC → Hindi PM-KISAN → mandi.
- [ ] Screenshot of the gate report (`v4_gate_final.md`).
- [ ] Zenodo DOI for the corrected v3 paper.
- [ ] Clinician/KVK reviewer call.
