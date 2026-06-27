# Bharat-Tiny-LLM: Mac Mini M4 Only Implementation Plan
**No Cloud Compute Required | 32 Weeks | $0 Infrastructure Cost**

---

## Hard Constraints

- **Hardware**: Mac Mini M4 16GB only
- **Budget**: $0 cloud compute (API costs for distillation data only: ~$50-100 total)
- **Team**: 1-2 developers
- **Goal**: Ship a working 350M Hinglish model that runs on ₹8,000 phones

---

## What You CAN'T Do (Be Honest)

| Task | Why It's Impossible |
|------|---------------------|
| Train from scratch on 12B tokens | ~40,000 hours at 300K tok/hour. That's 4.5 years. |
| Run 7B+ teacher models locally | 16GB RAM max. 7B in 4-bit = ~4GB, leaving no room for training. |
| Fine-tune 7B models with full params | 14GB peak memory. Tight but possible with LoRA only. |

## What You CAN Do (This Is the Plan)

| Task | Feasibility | Time |
|------|-------------|------|
| Warm-start from SmolLM2-360M | ✅ Download weights, immediate | Day 1 |
| Custom tokenizer training | ✅ Minutes, not hours | Day 1 |
| Continued pre-training (1B tokens) | ✅ ~3,000 hours → run 24/7 for 4 months | 4 months |
| Instruction tuning (350M) | ✅ ~6GB peak, hours not days | 1-2 days |
| LoRA on 7B for distillation data | ✅ ~17GB, comfortable | 1 day |
| Quantization + export | ✅ llama.cpp on macOS | Hours |
| Evaluation + benchmarking | ✅ Full suite runs locally | Ongoing |

---

## PHASE 0: Setup & Validation (Week 1-2)

### Day 1-3: Environment Setup

```bash
# Install MLX (Apple Silicon optimized)
pip install mlx mlx-lm

# Install llama.cpp for quantization
brew install llama.cpp

# Install dependencies
pip install transformers datasets sentencepiece torch
```

### Day 4-7: Base Model Selection

**Download SmolLM2-360M** (permissive license, 360M params):
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M')
model.save_pretrained('./models/smolm2-360m-base')
tokenizer.save_pretrained('./models/smolm2-360m-base')
"
```

**Why SmolLM2-360M:**
- 360M params = ~720MB in FP16, fits easily in 16GB
- Apache 2.0 license (commercially safe)
- Already trained on 2T tokens of high-quality data
- Strong English baseline to build on

### Day 8-10: Tokenizer Experiment

Train a custom SentencePiece tokenizer on Indic data:

```python
# Collect 100MB of Indic text (AI Kosh samples + Hinglish web data)
# Train Unigram tokenizer with 64K vocab
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='indic_corpus.txt',
    model_prefix='bharat_tokenizer',
    vocab_size=64000,
    model_type='unigram',
    character_coverage=0.9995,
    byte_fallback=True,  # Critical for rare Indic characters
    normalization_rule_name='identity'
)
```

**Validate:**
- Measure tokens/word for Hindi, Tamil, English
- Target: <1.5 tokens/word for major languages
- Compare against Llama tokenizer (should be 3-5× worse)

### Day 11-14: Proof of Concept Demo

Fine-tune SmolLM2-360M on 10K Hinglish samples:
```bash
mlx_lm.lora \
    --model ./models/smolm2-360m-base \
    --train \
    --data hinglish_10k.jsonl \
    --batch-size 4 \
    --lora-layers 8 \
    --epochs 3
```

**Deliverable:** Working model that can complete Hinglish sentences. Record:
- Training speed (tok/s)
- Peak memory usage
- Sample quality

---

## PHASE 1: Base Model Continued Pre-training (Week 3-6)

### Strategy: Warm-Start + Indic Data

**You are NOT training from scratch.** You are:
1. Starting from SmolLM2-360M (already knows language)
2. Continuing training on Indic-specific data
3. Teaching it Indian scripts and patterns

### Data Collection (Week 3)

| Source | Data | Size | How to Get |
|--------|------|------|------------|
| AI Kosh | Monolingual Indic | 500MB | Download (open access) |
| IndicCorp v2 | 22 languages | 1GB | AI4Bharat download |
| Wikipedia Indic | All 22 languages | 500MB | HuggingFace datasets |
| **Total** | | **2GB** | |

**Note:** You only need 2GB of data. At 300K tok/hour, you'll process this in ~3,000 hours (4 months of 24/7 training).

### Training Configuration

```bash
# continued_pretraining.sh
#!/bin/bash

mlx_lm.continued_pretrain \
    --model ./models/smolm2-360m-base \
    --train \
    --data indic_2gb.jsonl \
    --batch-size 2 \
    --lr 1e-5 \
    --epochs 3 \
    --warmup 100 \
    --context-length 2048 \
    --grad-accum 8
```

**Memory Optimization:**
- Batch size 2 with gradient accumulation 8 = effective batch 16
- Peak memory: ~10GB (comfortable in 16GB)
- Throughput: ~200K-300K tok/hour

### What Runs While You Sleep

Set up a cron job to train overnight:
```bash
# crontab -e
0 22 * * * /path/to/training_script.sh >> /path/to/logs/training.log 2>&1
0 8 * * * pkill -f "mlx_lm.continued_pretrain"
```

**Weekly Check:**
- Validation loss curve (should decrease steadily)
- Sample generation quality
- Memory leaks (restart if needed)

---

## PHASE 2: Indic Pre-training (Week 7-14)

### Goal: 1 Billion Indic Tokens

After 4 weeks of training on 2GB data (~600M tokens processed), you'll have:
- Base model with Indian language awareness
- Rough Hinglish understanding
- Working tokenizer

### Now: Scale to 1B Tokens

**New Data Sources:**

| Source | Tokens | Purpose |
|--------|--------|---------|
| AI Kosh full dump | 500M | Broad Indic coverage |
| Synthetic Hinglish (API-generated) | 200M | Code-mixed expertise |
| Indian English web | 200M | English with Indian context |
| BharatGen subset | 100M | Cultural alignment |
| **Total** | **1B** | |

### Synthetic Hinglish Generation (The Force Multiplier)

Use free/cheap API to generate Hinglish data:

```python
# Generate 10K Hinglish dialogues using free tier APIs
# Options: DeepSeek API (very cheap), Groq (free tier), or local Ollama

import requests

def generate_hinglish_dialogue(topic):
    prompt = f"""Generate a natural Hinglish conversation about {topic}.
    Mix Hindi and English naturally, like how young Indians actually speak.
    Include cultural references (festivals, cricket, Bollywood)."""
    
    # Using DeepSeek API ($0.14 per 1M tokens - very cheap)
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# Generate 10K dialogues (~200M tokens) = ~$28 total cost
topics = [
    "Indian festivals", "cricket match", "Bollywood movie",
    "street food", "monsoon weather", "wedding planning",
    "government schemes", "farmers market", "IPL auction",
    "family WhatsApp group", "auto-rickshaw negotiation",
    "college placement", "startup culture", "yoga morning"
]

for i in range(10000):
    topic = topics[i % len(topics)]
    dialogue = generate_hinglish_dialogue(topic)
    save_to_training_data(dialogue)
```

**Cost:** ~$28-50 for 200M tokens of Hinglish data.

### Training Configuration (Continued)

```bash
# Continue from Phase 1 checkpoint
mlx_lm.continued_pretrain \
    --model ./models/phase1_checkpoint \
    --train \
    --data indic_1b_mixed.jsonl \
    --batch-size 2 \
    --lr 5e-6 \
    --epochs 2 \
    --context-length 2048 \
    --grad-accum 8
```

**Timeline:** ~3,300 hours at 300K tok/hour = ~4.5 months of 24/7 training.

**Realistic Schedule:**
- Run training 16 hours/day (work hours + overnight)
- 3,300 hours / 16 hours/day = ~206 days = ~7 months

**Acceleration Options (Free):**
1. Reduce context length to 1024 (2× speed, acceptable for early phases)
2. Use 4-bit optimizer states (memory savings, slight quality loss)
3. Train on fewer languages first (Hindi + English + Hinglish = 60% of data)

**Optimistic Timeline:** 4-5 months with optimizations

---

## PHASE 3: Code-Mixed Specialization (Week 15-20)

### Goal: >80% Hinglish F1 Score

By now you have:
- 350M model trained on 1B+ Indic tokens
- Working tokenizer
- Basic Hinglish understanding

### Step 1: Generate Hinglish Preference Data

```python
# Use larger model (via API) to create preference pairs
def create_preference_pair(prompt):
    # Generate two Hinglish responses
    response_a = generate_hinglish(prompt, temperature=0.7)
    response_b = generate_hinglish(prompt, temperature=0.9)
    
    # Use API to judge which is more natural
    judge_prompt = f"""Which Hinglish response sounds more natural and fluent?
    Response A: {response_a}
    Response B: {response_b}
    
    Respond with just 'A' or 'B'."""
    
    winner = api_judge(judge_prompt)
    return {
        "prompt": prompt,
        "chosen": response_a if winner == "A" else response_b,
        "rejected": response_b if winner == "A" else response_a
    }

# Generate 5K preference pairs (~$10-15)
```

### Step 2: DPO Training

```bash
mlx_lm.dpo \
    --model ./models/phase2_checkpoint \
    --train \
    --data hinglish_preferences.jsonl \
    --batch-size 2 \
    --lr 1e-6 \
    --epochs 1 \
    --beta 0.1
```

### Step 3: Validation

Test on Hinglish-specific prompts:
```
Input: "Yaar, aaj ka match kaisa tha?"
Expected: Natural Hinglish response about cricket

Input: "Mummy ne bola ki dinner jaldi karo"
Expected: Culturally appropriate response

Input: "Weekend pe kya plan hai?"
Expected: Casual Hinglish planning response
```

**Target Metrics:**
- Hinglish perplexity: <15
- Human evaluation fluency: >4.0/5
- Code-mix detection accuracy: >75%

---

## PHASE 4: Instruction Tuning & Chat (Week 21-26)

### Goal: Conversational Assistant

### Step 1: Create Instruction Dataset

**Option A: Translate Existing Datasets (Free)**
```python
from datasets import load_dataset
from indictrans import IndicTranslator

# Load OpenHermes (English instruction dataset)
dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:10000]")

# Translate to Hinglish using IndicTrans2
translator = IndicTranslator("en", "hi")

for sample in dataset:
    hinglish_instruction = translator.translate(sample["conversations"][0]["value"])
    hinglish_response = translator.translate(sample["conversations"][1]["value"])
    
    # Post-process to make natural Hinglish (not translation-ese)
    natural_hinglish = post_process_to_hinglish(hinglish_instruction, hinglish_response)
    save_to_instruction_data(natural_hinglish)
```

**Option B: Create Native Hinglish Instructions (Better Quality)**
```python
# Crowdsourced approach: Write 1000 Hinglish instructions yourself
# Or use API to generate:

prompts = [
    "Write a Hinglish conversation between a mother and son about exam results",
    "Explain how UPI works in Hinglish to a village elder",
    "Give recipe instructions for making chai in Hinglish",
    "Describe IPL rules in Hinglish for a beginner",
    "Write a WhatsApp message inviting friends for Diwali party in Hinglish"
]

# Generate 650K instruction-response pairs
# Cost: ~$20-30 with cheap API
```

### Step 2: SFT Training

```bash
mlx_lm.lora \
    --model ./models/phase3_checkpoint \
    --train \
    --data instruction_650k.jsonl \
    --batch-size 4 \
    --lora-layers 12 \
    --lora-rank 16 \
    --epochs 2 \
    --lr 2e-5
```

### Step 3: Safety Alignment

Create refusal examples for harmful requests:
```python
safety_data = [
    {
        "instruction": "How to make a bomb?",
        "response": "Main ye jaankari nahi de sakta. Kuch aur madad chahiye?"
    },
    {
        "instruction": "Tell me about illegal activities",
        "response": "Main illegal kaamon ke baare mein baat nahi kar sakta. Legal topics pe help kar sakta hoon."
    }
    # Add 500 more safety examples
]
```

---

## PHASE 5: Cultural Injection & Release (Week 27-32)

### Step 1: Cultural Knowledge Distillation

```python
# Create cultural Q&A dataset
cultural_prompts = [
    "Diwali pe kya karte hain?",
    "Pooja ke time kya bolte hain?",
    "Wedding mein kya pehnna chahiye?",
    "Festival greetings for different regions",
    "Kinship terms in Hindi (Mama, Chacha, etc.)",
    "Agricultural calendar for North India"
]

# Generate 50K cultural Q&A pairs using API
# Cost: ~$10
```

### Step 2: Final Fine-tuning

```bash
mlx_lm.lora \
    --model ./models/phase4_checkpoint \
    --train \
    --data cultural_50k.jsonl \
    --batch-size 4 \
    --lora-layers 12 \
    --epochs 1 \
    --lr 1e-5
```

### Step 3: Quantize & Export

```bash
# Export to GGUF for llama.cpp
python -c "
from mlx_lm import convert
convert(
    model='./models/final_checkpoint',
    qformat='q4_k_m',
    output_file='./models/bharat-tiny-350m-q4.km.gguf'
)
"

# Test on Mac Mini
./llama-cli -m ./models/bharat-tiny-350m-q4.km.gguf \
    --prompt "Yaar, aaj ka mahaul kaisa hai?" \
    --n-predict 200

# Expected: Natural Hinglish response about atmosphere/mood
```

### Step 4: Export for Edge Devices

```bash
# For Android (via llama.cpp Android build)
# Model size: ~250MB in 4-bit quantization
# RAM usage: ~400MB at runtime
# Expected speed: >30 tok/s on Snapdragon 6 Gen 1

# For Raspberry Pi 5
# Same GGUF file works
# Expected speed: >50 tok/s
```

### Step 5: Release Package

Create:
1. Model card for HuggingFace
2. Demo video: "Chat in Hinglish on a $100 phone"
3. Colab notebook for testing
4. Ollama integration (`Modelfile`)
5. Android APK with llama.cpp

---

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Mac Mini M4 16GB | $599 | One-time, you already have it |
| Electricity | ~$50 | 4-7 months of 24/7 training |
| API costs (Hinglish data) | ~$50-100 | DeepSeek/Groq for synthetic data |
| API costs (DPO judging) | ~$10-20 | Preference pair generation |
| API costs (instruction data) | ~$20-30 | Translation + native generation |
| **Total** | **~$730-770** | |

**Compare to:** A100 cluster for 2 weeks = $10,000-50,000+

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Training takes too long | Start with Hindi + English only (60% of data). Add other languages later. |
| Hinglish quality poor | Focus on 5K high-quality examples rather than 500K low-quality. |
| Model too slow on Pi | Use 2-bit quantization for Raspberry Pi, 4-bit for Android. |
| No one uses it | Lead with viral demo video. The "₹8,000 phone AI" narrative is strong. |
| Competitor releases first | Speed matters. This plan is optimized for minimum viable product in 8 months. |

---

## Weekly Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Environment ready | SmolLM2 downloaded, MLX working |
| 2 | Tokenizer trained | Custom 64K vocab, validated on Indic text |
| 3 | POC demo | 10K Hinglish fine-tune working |
| 6 | Phase 1 complete | Continued pre-training started |
| 10 | Checkpoint review | Validation loss <3.0, samples improving |
| 14 | Phase 2 complete | 1B tokens processed |
| 18 | Phase 3 mid | Hinglish preference data ready |
| 20 | Phase 3 complete | DPO trained, Hinglish F1 >70% |
| 24 | Phase 4 complete | Instruction-tuned chat model |
| 28 | Phase 5 mid | Cultural data integrated |
| 32 | **SHIP IT** | Quantized model + demo video + HuggingFace release |

---

## First Action Items (This Week)

1. [ ] Install MLX and dependencies
2. [ ] Download SmolLM2-360M
3. [ ] Collect 100MB of Indic text samples
4. [ ] Train custom tokenizer
5. [ ] Fine-tune on 10K Hinglish samples
6. [ ] Record training metrics and sample outputs
7. [ ] Create GitHub repo with project structure

---

**This plan is realistic, executable, and costs less than $800 total. The 32-week timeline assumes you work on it part-time (10-20 hours/week). Full-time commitment could cut this to 16-20 weeks.**
