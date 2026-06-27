# Bharat-Tiny-LLM: BRAHMI Architecture — Complete Build & GTM Plan
**By Eulogik** | *The First Language Model Built for India's Linguistic Reality*

---

## 1. Executive Summary

Bharat-Tiny-LLM is not a smaller version of existing cloud models. It is a **new species of language model** — the BRAHMI architecture — designed from first principles for India's unique linguistic landscape: 22 scheduled languages, seamless code-mixing (Hinglish, Tanglish), Brahmic script compositionality, and cultural grounding that runs natively on edge devices.

**The North Star**: Create a model so culturally intimate and linguistically native that an Indian farmer on a ₹8,000 phone can have a natural conversation in his dialect, and a developer on a Mac Mini M4 can train it on her desk.

**For Eulogik**: This is not just a model. It is a **movement** — "India's first truly native AI" — designed for national pride, developer love, and viral cultural resonance.

---

## 2. Core Philosophy

### 2.1 The Four Rejections

| Western Assumption | BRAHMI Rejection | Our Reality |
|-------------------|------------------|-------------|
| Text is linear Unicode | **Script is visual composition** | Brahmic scripts are 2D glyph systems |
| Languages are separate | **Code-mixing is native blending** | Hinglish is not Hindi + English; it is a third language |
| Knowledge is static weights | **Culture is dynamic generation** | Indian context varies by festival, region, kinship |
| One model fits all dialects | **Models should evolve per user** | India has 1,600+ dialects; static models are culturally dead |

### 2.2 Design Principles

1. **Edge-First**: If it doesn't run on a Raspberry Pi 5 or Mac Mini M4 16GB, it doesn't ship.
2. **Cultural-Native**: Not translated. Not adapted. Born Indian.
3. **Open-Core**: Weights open (Apache 2.0). Ecosystem open. Premium layers commercial.
4. **Living Model**: It gets smarter with every Indian who uses it.
5. **Paper-Worthy**: Every component must be a novel contribution to AI science.

---

## 3. Architecture: BRAHMI

### 3.1 System Overview

BRAHMI (Brahmic Representation Architecture for Hypernetworked Multilingual Intelligence) consists of four interlocking innovations:

```
User Input (Hinglish/Devanagari/Tamil/Roman)
        │
        ▼
┌─────────────────────────────────────┐
│  BNC: Brahmic Neural Codec          │
│  (Vision Transformer for glyphs)    │
│  → Syllabic latent tokens           │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  DLR: Dynamic Language Router       │
│  (Bilingual brain-inspired gating)  │
│  → Language activation vectors      │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  CKH: Cultural Knowledge Hypernet   │
│  (Generates contextual adapters)    │
│  → Cultural LoRA weights            │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Core Transformer (300M params)    │
│  + EFP: Edge Federated Personality │
│  → Personalized response           │
└─────────────────────────────────────┘
```

### 3.2 Component Specifications

#### BNC — Brahmic Neural Codec

| Attribute | Specification |
|-----------|---------------|
| **Type** | 4-layer Vision Transformer (ViT) |
| **Parameters** | 3M |
| **Input** | Rendered akshara glyphs (32×32 grayscale) |
| **Output** | 256-dim syllabic latent tokens |
| **Vocab Size** | 64K glyph tokens |
| **Key Feature** | Zero-shot script transfer (Devanagari ↔ Gujarati ↔ Bengali) |
| **Efficiency Gain** | 0.9 tokens/syllable vs. 3.5 for BPE (4× context window boost) |

**Why it matters**: For the first time, an LLM "reads" Indian script the way humans do — as visual syllabic units, not as broken Unicode sequences.

#### DLR — Dynamic Language Router

| Attribute | Specification |
|-----------|---------------|
| **Type** | Gating network + attention head reweighting |
| **Parameters** | 2M |
| **Mechanism** | Continuous language activation vectors (not binary switches) |
| **Slots** | 8: Hindi, English, Tamil, Telugu, Bengali, Marathi, Hinglish, Tanglish |
| **Key Feature** | Native code-mixed blending without explicit code-mixed training data |
| **Inspiration** | Bilingual Interactive Activation (BIA) model from cognitive neuroscience |

**Why it matters**: The model doesn't "detect" Hinglish and switch modes. It maintains overlapping Hindi-English neural representations and blends them dynamically, just like a bilingual human brain.

#### CKH — Cultural Knowledge Hypernetwork

| Attribute | Specification |
|-----------|---------------|
| **Type** | Hypernetwork (network that generates networks) |
| **Parameters** | 50M (hypernetwork) + 5M generated adapters |
| **Input** | Cultural context embedding (festival, relation, region, tone) |
| **Output** | LoRA adapter weights injected into every transformer layer |
| **Knowledge Base** | 50,000-node differentiable cultural knowledge graph |
| **Key Feature** | Culture is not retrieved; it is *embodied* through generated weights |

**Why it matters**: Ask "What to say at a funeral?" and the model doesn't search a database. It generates weights that make the model *behave* like a culturally grounded Indian would.

#### EFP — Edge Federated Personalization

| Attribute | Specification |
|-----------|---------------|
| **Type** | Federated meta-learning (MAML initialization) |
| **Mechanism** | On-device dialect adaptation + encrypted gradient upload |
| **Key Feature** | Model evolves per user while preserving privacy |
| **Aggregation** | Differential privacy + secure multi-party computation |
| **Meta-Learning** | 10–20 gradient steps to adapt to any new dialect |

**Why it matters**: This is the first "living language model." A Rajasthani farmer's instance becomes Rajasthani-specialized. A Chennai student's becomes Tanglish-fluent. All contribute back to the global model.

### 3.3 Core Transformer

| Attribute | Specification |
|-----------|---------------|
| **Parameters** | 300M active (150M per forward pass via conditional computation) |
| **Architecture** | Decoder-only, 24 layers, 12 heads, 1024 dim |
| **Attention** | Grouped Query Attention (GQA) + DLR head routing |
| **Activation** | SwiGLU |
| **Position Encoding** | RoPE with YaRN (8K base → 32K extended) |
| **Normalization** | RMSNorm pre-normalization |
| **Context Window** | 8K base / 32K extended |
| **Active Parameters** | 150M per forward pass (language heads conditionally loaded) |

---

## 4. Development Phases

### Phase 0: Baby BRAHMI — Architecture Prototype

**Objective**: Prove BNC + DLR works. Get a working demo in 4 weeks. Establish scientific priority.

**Model Spec**:
- 100M parameters
- 8 layers, 8 heads, 512 dim
- BNC: 2-layer ViT, 48K glyph vocab
- DLR: 4 language slots (Hindi, English, Hinglish, Tamil), 1M gating
- CKH: Disabled
- EFP: Disabled

**Training**:
- Data: 1B tokens (40% Hindi, 30% English, 20% Hinglish, 10% Tamil)
- Platform: Mac Mini M4 16GB (MLX framework)
- Duration: Entirely local, ~2 weeks

**Deliverables**:
- Working 100M model generating coherent text in 4 languages
- BNC token efficiency validation (>4× improvement over BPE)
- DLR zero-shot Hinglish demonstration
- arXiv paper: *"Baby BRAHMI: A 100M Proof-of-Concept for Brahmic Script Understanding"*
- **Viral asset**: Video demo — "This 100M model reads Hindi like a human, not like a computer"

**Success Criteria**:
- BNC reconstruction accuracy >99%
- Hinglish perplexity lower than Llama-3.2-1B fine-tuned on same data
- Inference >100 tok/s on Mac Mini M4

---

### Phase 1: Core BRAHMI — The Full Base Model

**Objective**: Build the 350M parameter Bharat-Tiny-LLM v1.0. The first native Indic edge model.

**Model Spec**:
- 350M parameters
- 24 layers, 12 heads, 1024 dim
- BNC: 4-layer ViT, 64K glyph vocab, zero-shot script transfer
- DLR: 8 language slots (22 scheduled + English + code-mixed blends), 2M gating
- CKH: 50M hypernetwork, 5K cultural concept graph
- EFP: MAML initialization only

**Training Pipeline**:

| Stage | Tokens | Data | Loss | Platform |
|-------|--------|------|------|----------|
| **Distillation** | 0–2B | English Wikipedia + instructions | KL divergence vs. SmolLM2-360M | Cloud burst (4×A100 or BharatGen TIHub) |
| **Indic Pre-training** | 2B–6B | AI Kosh monolingual (22 languages) | Next-token prediction | Cloud burst |
| **Code-Mixed & Cultural** | 6B–10B | Synthetic Hinglish + BharatGen + cultural corpus | Next-token + DLR consistency | Cloud burst |
| **Instruction Tuning** | 10B–12B | Indic instruction data (native + translated) | SFT + DPO preference | Mac Mini M4 |
| **CKH Training** | — | Cultural Q&A pairs | Hypernetwork adapter generation | Mac Mini M4 |

**Mac Mini M4 Role**:
- Architecture ablations and tokenizer experiments
- Instruction tuning and DPO (full fine-tuning fits in 16GB)
- CKH hypernetwork training
- Quantization and export (GGML, CoreML, TFLite)
- Evaluation and benchmarking

**Deliverables**:
- 350M base model (Apache 2.0)
- 4-bit quantized edge variants (250MB, 400MB, 800MB)
- BharatTiny-Bench evaluation suite
- Hugging Face model card + Ollama integration
- **Paper**: *"BRAHMI: A Native Architecture for India's Linguistic Reality"* (ACL/EMNLP)
- **Viral asset**: Side-by-side video — BRAHMI vs. GPT-4o on Hinglish cultural questions

**Success Criteria**:
- IndicQA >70%
- Hinglish F1 >80%
- >50 tok/s on Raspberry Pi 5
- >120 tok/s on Mac Mini M4
- Cultural commonsense >85%

---

### Phase 2: BRAHMI Chat — Conversational & Cultural Depth

**Objective**: Convert the base model into a warm, culturally aware conversational AI.

**Training**:
- 650K high-quality instruction samples (following Navarasa scale but native quality)
- Multi-turn Hinglish conversations with cultural context injection
- Safety-aligned refusals for harmful requests in Indic contexts
- RLAIF (Reinforcement Learning from AI Feedback) for code-mixed naturalness

**CKH Expansion**:
- Expand cultural knowledge graph to 20,000 nodes
- Add domain-specific cultural contexts: legal, medical, agricultural, educational
- Regional cultural adapters: North Indian, South Indian, East Indian, North-East Indian

**Deliverables**:
- Bharat-Tiny-LLM-Chat-v1.0
- Cultural adapter packs (downloadable LoRA sets)
- Demo app: "Chat with Bharat" — web + Android APK
- **Paper**: *"Cultural Knowledge Hypernetworks for Contextual AI"* (FAccT/NAACL)

**Success Criteria**:
- Human-evaluated Hinglish fluency >4.5/5
- Cultural appropriateness score >90%
- Multi-turn coherence across 10+ turns

---

### Phase 3: BRAHMI Live — Federated Personalization

**Objective**: Activate the living model. Deploy federated learning network.

**EFP Activation**:
- Release federated learning client for Mac Mini, Raspberry Pi, and Android
- Central aggregation server (Eulogik-hosted, privacy-preserving)
- Dialect Genome Server: collects encrypted gradient deltas, produces global updates

**Dialect Coverage**:
- Tier 1 (Excellent): Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam
- Tier 2 (Good): Remaining 14 scheduled languages
- Tier 3 (Emerging): Major dialects (Bhojpuri, Awadhi, Haryanvi, Rajasthani, etc.)

**Developer Ecosystem**:
- "Bharat Tiny Fellows" program: 50 university labs, each owning one language/dialect
- Federated adapter merging protocol
- Mac Mini M4 as the "federation node" for local communities

**Deliverables**:
- Bharat-Tiny-LLM-Live-v2.0
- Federated learning SDK
- Dialect Genome dataset (anonymized, aggregated)
- **Paper**: *"Dialect Darwinism: Federated Meta-Learning for Linguistic Diversity"* (NeurIPS/FL)

**Success Criteria**:
- 1,000+ active federated devices
- +15% accuracy improvement on user dialect after 20 turns
- Zero privacy breaches (audited by third party)

---

### Phase 4: BRAHMI Vision — Multimodal Edge

**Objective**: Extend BRAHMI to see and understand documents in Indic scripts.

**Architecture**:
- 500M parameter vision-language model
- BNC extended to document images (receipts, forms, newspapers)
- Integration with Indic-Doc-VLM pipeline
- On-device RAG with Indian government document stores

**Use Cases**:
- Farmer reads a pesticide label in Hindi → gets safety instructions in Hinglish
- Student scans a math problem in Tamil → gets step-by-step explanation in Tanglish
- Government kiosk reads an Aadhaar form → fills it via voice in local dialect

**Deliverables**:
- Bharat-Tiny-VLM-500M
- Document understanding benchmark (IndicDoc-Bench)
- **Paper**: *"Vision-Language Models for Brahmic Script Documents"* (CVPR/ICCV)

---

### Phase 5: BRAHMI Platform — The Ecosystem Layer

**Objective**: BRAHMI becomes infrastructure, not just a model.

**Platform Components**:
- **BRAHMI Hub**: Model registry, adapter marketplace, cultural knowledge graph editor
- **BRAHMI Studio**: No-code fine-tuning on Mac Mini M4 for Indian startups
- **BRAHMI Voice**: End-to-end voice assistant (ASR → BRAHMI → TTS) via BHASHINI integration
- **BRAHMI API**: Hosted fallback for enterprises (premium tier)

**Standardization**:
- BNC becomes the default tokenizer for all future Indic NLP research
- DLR gating adopted as a standard for code-mixed models
- Cultural hypernetworks become a subfield of AI alignment

**Deliverables**:
- BRAHMI Platform v1.0
- 10+ Indian startups building on BRAHMI
- Integration with BHASHINI government infrastructure
- **Capstone Paper**: *"BRAHMI: A New Paradigm for Linguistically Diverse Edge AI"* (Nature Machine Intelligence)

---

## 5. Data Strategy

### 5.1 Data Sources

| Source | Type | Usage | Status |
|--------|------|-------|--------|
| **AI Kosh** | Government dataset repository | Base pre-training (12B+ tokens) | Open access |
| **BharatGen Corpus** | Sovereign text corpus | Continued pre-training, cultural alignment | Government partnership |
| **BHASHINI / Project Vaani** | Speech + transcribed text | Phonetic Hinglish, ASR-TTS integration | Open access |
| **AI4Bharat IndicNLP** | Monolingual corpora | Foundation vocabulary, 10 languages | Open access |
| **IndicCorp v2** | Large-scale monolingual | Pre-training for 22 languages | Open access |
| **PolyWhisper** | Multilingual speech | Cross-modal text extraction | Internal/Eulogik |
| **Indic-Doc-VLM** | Document images | OCR text for low-resource languages | Internal/Eulogik |

### 5.2 Synthetic Data Generation

**Code-Mixed Corpus**:
- Generate 500M–1B tokens of naturalistic Hinglish, Tanglish, Banglish, etc.
- Use equivalence constraint theory (Poplack, 1980) for syntactic validity
- Balance across 6 major code-mixing pairs

**Cultural Knowledge**:
- Generate Q&A on festivals, governance, agriculture, Bollywood, cricket, regional cuisines
- Distill from larger models (Krutrim-2, Sarvam-30B) but verify against government sources
- Create "Cultural Stress Tests" — edge cases where Western models fail

**Conversational AI**:
- Translate + adapt high-quality English instruction datasets (OpenHermes, UltraChat) using IndicTrans2
- Create native Indic instruction data via "Bharat Data Fellows" crowdsourcing
- Safety data: refusal patterns for harmful requests in Indic contexts

### 5.3 Tokenization (BNC)

- Custom Unigram SentencePiece trained on 22-language + code-mixed corpus
- Average 1.2–1.5 tokens per word for major Indic languages
- Special tokens: `<|hindi|>`, `<|tamil|>`, `<|hinglish|>`, `<|tanglish|>` for explicit routing
- Zero-Shot Script Transfer: Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Odia, Tamil, Telugu share visual features → BNC generalizes

---

## 6. Evaluation & Benchmarking

### 6.1 Existing Benchmarks

| Benchmark | Target | Current SOTA |
|-----------|--------|--------------|
| **IndicQA** | >70% | Sarvam-105B ~68% |
| **BharatBench** | Top-3 | Sarvam-105B (top-3 on 5/7 tasks) |
| **CodeMixBench (CM-MMLU)** | >60% | GPT-4o ~72% |
| **GLUECoS** | NLI/QA/SA | Fine-tuned models beat zero-shot LLMs |

### 6.2 New Benchmarks (BharatTiny-Bench)

Define benchmarks that favor tiny, culturally-aware models:

**BharatTiny-Bench**:
- IndicQA subset optimized for 500M models (shorter contexts, common knowledge)
- Hinglish Conversational Fluency (human-evaluated, 1,000 dialogues)
- Cultural Commonsense (e.g., funeral etiquette, festival greetings, kinship terms)

**Edge-Indic-Perf**:
- Tokens/second on Raspberry Pi 5, Android (Snapdragon 6 Gen 1), Mac Mini M4
- Energy consumption per 1K tokens
- Time-to-first-token (TTFT) for Hinglish prompts

**Code-Switch Robustness**:
- Auto-detect language mix ratio (30% Hindi / 70% English vs. 70% Hindi / 30% English)
- Maintain coherence across 5+ turn conversations with escalating code-mixing

**Zero-Shot Script Transfer**:
- Devanagari → Gujarati accuracy
- Roman Hinglish → Devanagari Hindi translation without parallel data

---

## 7. Go-To-Market Strategy

### 7.1 Positioning

**The Category**: Bharat-Tiny-LLM does not compete with Sarvam or Krutrim. It creates a new category: **"Indic Edge-Native AI."**

**Positioning Statement**:
> *"Bharat-Tiny-LLM is the first language model built from the ground up for India's linguistic reality — 22 scheduled languages, seamless Hinglish code-switching, and cultural warmth — at a size that runs on a ₹8,000 phone without internet."*

### 7.2 Target Audiences

| Segment | Pain Point | BRAHMI Solution |
|---------|------------|-----------------|
| **Indian Developers** | Building vernacular chatbots requires cloud APIs or bloated 7B models | 350M model that "just works" for Hinglish, runs locally |
| **Agritech Startups** | Farmers have no internet, speak dialects, need offline AI | On-device agricultural advisor in Bhojpuri-Hinglish |
| **EdTech Startups** | Low-bandwidth tutoring for Bharat users | Offline tutor that speaks the student's language |
| **Government (BHASHINI)** | Sovereign AI for kiosks and rural centers | Privacy-first, offline, culturally appropriate |
| **Indian Diaspora** | Children losing touch with native language | Conversational practice in mother tongue + Hinglish |
| **AI Researchers** | No benchmark for tiny Indic models | BharatTiny-Bench, novel architecture, open weights |

### 7.3 Distribution Channels

**Primary**:
- **Hugging Face**: Model hub, 4-bit/8-bit/FP16 variants, model cards
- **AI Kosh**: Government repository for sovereign AI compliance
- **Ollama / LM Studio**: One-command install for Mac Mini users
- **GitHub**: Full training code, BNC renderer, evaluation suite

**Secondary**:
- **BHASHINI Integration**: Default LLM for their open-source handheld device
- **Android APK**: Pre-bundled with llama.cpp for offline chat
- **Raspberry Pi Image**: Pre-configured OS image with BRAHMI + voice pipeline
- **BharatGen TIHub**: Official partnership for compute and distribution

### 7.4 Pricing & Commercial Model

**Open Source (Apache 2.0)**:
- Base 350M model weights
- BNC tokenizer + training code
- BharatTiny-Bench evaluation suite
- Mac Mini M4 training recipes
- Federated learning SDK

**Premium / Commercial**:
- **Bharat-Tiny-Enterprise**: Domain-specific fine-tuning (legal, medical, agricultural) with RAG-optimized embeddings
- **Bharat-Tiny-Voice**: End-to-end voice assistant (ASR → BRAHMI → TTS) via BHASHINI
- **Bharat-Tiny-API**: Hosted cloud fallback for enterprises (per-token pricing, but edge is the hero)
- **BRAHMI Studio**: No-code fine-tuning platform for Indian startups (SaaS)
- **Cultural Adapter Packs**: Premium cultural knowledge graphs for specific regions/communities

---

## 8. Virality & Popularity Engine for Eulogik

### 8.1 The Narrative Arc

**The Story**: Eulogik is not a company making a model. Eulogik is a **movement** reclaiming AI for Bharat.

**Chapter 1: The Provocation**
> *"Why do India's AI models speak English better than Hindi? Why does a farmer need a ₹50,000 phone and a 4G tower to ask about weather? Eulogik is building India's first AI that speaks your language, your dialect, your Hinglish — on the phone you already own."*

**Chapter 2: The Proof**
> Release Baby BRAHMI with a video showing the 100M model reading Devanagari as glyphs, not characters. Contrast with GPT-4o breaking Hindi words into 5 tokens each.

**Chapter 3: The Cultural Moment**
> Release Core BRAHMI during a major Indian festival (Diwali, Holi, Pongal). Demonstrate CKH generating culturally appropriate greetings that change based on region and relationship.

**Chapter 4: The Grassroots Revolution**
> Launch "Bharat Tiny Fellows." Show 50 university labs across India training their own dialect adapters on Mac Mini M4s. Visualize the Dialect Genome growing.

**Chapter 5: The National Pride**
> BRAHMI runs on BHASHINI government kiosks. National news coverage. Ministerial mentions. "India's AI sovereignty starts at the edge."

### 8.2 Viral Content Strategy

**Video Series: "BRAHMI vs. The World"**

| Episode | Hook | Platform |
|---------|------|----------|
| **Ep 1: The Glyph Test** | BRAHMI reads *क्‍ष* as one token. GPT-4o reads it as five. | YouTube, Twitter/X, LinkedIn |
| **Ep 2: The Hinglish Turing Test** | Can you tell if BRAHMI or your cousin wrote this WhatsApp message? | Instagram Reels, YouTube Shorts |
| **Ep 3: The ₹8,000 Phone Challenge** | BRAHMI runs on a $100 Android phone. GPT-4o needs cloud + $1,000 phone. | Twitter/X, LinkedIn |
| **Ep 4: The Cultural Test** | Ask both models "What do I say when my neighbor's father passes away?" BRAHMI knows the regional variation. GPT-4o gives generic condolences. | YouTube, LinkedIn |
| **Ep 5: The Dialect Experiment** | Give BRAHMI to a Rajasthani farmer. After 20 chats, it speaks his dialect. | Documentary-style, all platforms |

**Interactive Demos**:
- **"Hinglish or Human?"** — Web game where users guess if a message was written by BRAHMI or a human. Share score. Viral loop.
- **"My Dialect, My AI"** — Users upload 20 lines of their dialect. BRAHMI adapts and generates a poem in their dialect. Shareable output.
- **"BRAHMI Cultural Compass"** — Input a situation (festival, funeral, wedding). BRAHMI generates the culturally appropriate response for your region. Shareable.

**Meme Strategy**:
- "When GPT-4o thinks 'kya' is 3 tokens but BRAHMI knows it's 1 syllable"
- "My grandmother's AI speaks better Hinglish than my NRI cousin"
- "BRAHMI runs on a ₹8,000 phone. Your AI runs on a ₹8,000/month cloud bill."

### 8.3 Community Building

**"Bharat Tiny Fellows"**:
- 50 university labs across India
- Each lab gets a Mac Mini M4 (or uses their own)
- Each lab owns one language/dialect
- Monthly virtual meetups, leaderboard of dialect adapters
- Top contributors get speaking slots at Eulogik's annual "BRAHMI Summit"

**Discord / Telegram Community**:
- #hinglish-chat: Share funny Hinglish conversations with BRAHMI
- #dialect-corner: Upload dialect samples, request adapters
- #cultural-tests: Stress-test CKH with obscure cultural questions
- #mac-mini-heroes: Share training logs, benchmark results

**Hackathons**:
- "BRAHMI Buildathon" — 48-hour hackathon for building apps on BRAHMI
- Categories: Agritech, EdTech, GovTech, HealthTech, Entertainment
- Prize: Mac Mini M4 + BRAHMI swag + Eulogik internship

### 8.4 PR & Media Strategy

**The Pitch**: *"India's first AI built for the Next Billion Users"*

**Tier 1 Targets**:
- **National News**: NDTV, India Today, The Hindu, Indian Express — "The ₹100 Phone AI"
- **Tech Media**: TechCrunch India, YourStory, Inc42 — "Eulogik's BRAHMI challenges the cloud AI giants"
- **Government**: Press Information Bureau, MyGov — "Sovereign AI for Digital India"
- **Academic**: ACL, EMNLP, NeurIPS — Paper acceptance generates credibility

**Tier 2 Targets**:
- **YouTube Influencers**: Tech reviewers (Beebom, Trakin Tech) — "I ran an AI on a $100 phone"
- **Podcasts**: Indian tech podcasts (IVM, The Seen and the Unseen) — "The philosophy behind BRAHMI"
- **Regional Media**: Hindi, Tamil, Telugu news channels — "AI in your language"

**The Stunt**:
- Install BRAHMI on 100 Raspberry Pi devices.
- Distribute to 100 villages across India.
- Document the stories. Create a documentary: *"100 Villages, 100 Languages, 1 AI"*
- Release on YouTube. Pitch to Netflix/Prime Video India.

### 8.5 Partnership Strategy

**Strategic Partners**:

| Partner | Value | Eulogik Value |
|---------|-------|---------------|
| **BharatGen / MeitY** | Compute access, government legitimacy | Edge deployment for sovereign AI |
| **BHASHINI** | ASR/TTS infrastructure, government reach | LLM layer for their devices |
| **Sarvam AI** | Cloud ecosystem, startup network | Edge complement to their cloud models |
| **IIT Bombay (TIHub)** | Research talent, compute | Industry partnership, student pipeline |
| **AI4Bharat** | Datasets, research credibility | Real-world deployment platform |
| **Agritech Startups (DeHaat, etc.)** | Real users, domain data | Offline AI for farmers |
| **EdTech Startups** | Distribution to students | Low-bandwidth tutoring |

---

## 9. Team & Execution Structure

### 9.1 Core Team (Eulogik Internal)

| Role | Responsibility |
|------|----------------|
| **Chief Architect** | BRAHMI system design, BNC + DLR + CKH integration |
| **Indic NLP Lead** | Data curation, tokenizer design, benchmark creation |
| **Edge Engineering Lead** | MLX optimization, quantization, Raspberry Pi deployment |
| **Cultural AI Lead** | CKH knowledge graph, cultural data generation, safety |
| **Federated Learning Lead** | EFP architecture, privacy, aggregation server |
| **Developer Advocate** | Community, docs, tutorials, hackathons |
| **GTM Lead** | Partnerships, PR, viral strategy, Eulogik brand |

### 9.2 Extended Team (Community)

| Role | Source | Incentive |
|------|--------|-----------|
| **Bharat Tiny Fellows** | 50 university labs | Mac Mini M4 grants, co-authorship, summit speaking slots |
| **Dialect Contributors** | Crowdsourced | Swag, recognition in model card, early access |
| **Cultural Advisors** | Anthropologists, linguists | Co-authorship, advisory fees |
| **Beta Developers** | Indian startup ecosystem | Free API credits, premium features, co-marketing |

### 9.3 Execution Rhythm

**Weekly**:
- Architecture standup (BNC/DLR/CKH progress)
- Data pipeline review
- Community engagement check

**Bi-weekly**:
- Bharat Tiny Fellows sync
- Benchmark evaluation run
- Viral content planning

**Monthly**:
- Model release candidate review
- Partnership check-in
- PR/media strategy review
- Paper writing sprint

**Quarterly**:
- Major model release (v0.1 → v1.0 → v2.0)
- BharatTiny-Bench update
- Hackathon / summit
- Board/steering review

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Large models get cheaper** | Position on privacy + offline + latency. A 105B model cannot run on a ₹8,000 phone even if inference is free. |
| **BharatGen releases a tiny model** | BharatGen is focused on sovereign cloud models (2.9B+). Collaborate — offer to be their "edge deployment partner." |
| **Hinglish quality doesn't hit 80%** | Use RLAIF with human-in-the-loop. If 80% is unreachable at 350M, release a 500M "Hinglish Pro" variant while keeping 350M for monolingual tasks. |
| **Mac Mini M4 can't train from scratch** | Use warm-start + distillation for base. Mac Mini is for iteration, instruction tuning, and quantization. Cloud burst only for 12B-token pre-training. |
| **Low-resource language quality poor** | Use transfer learning from related high-resource languages. Accept tiered quality (Tier 1: 8 languages excellent; Tier 2: 14 languages good). |
| **Community doesn't form** | Lead with viral content and hackathons. "Bharat Tiny Fellows" creates ownership. Mac Mini M4 grants create tangible commitment. |
| **Competitor copies BRAHMI** | Open-source everything. The value is in the ecosystem and the Eulogik brand, not the weights. First-mover advantage in community and partnerships. |

---

## 11. Success Metrics

### 11.1 Technical Metrics

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|---------|
| **Parameters** | 100M | 350M | 350M | 350M | 500M |
| **IndicQA** | — | >70% | >72% | >75% | >78% |
| **Hinglish F1** | >60% | >80% | >85% | >88% | >90% |
| **Cultural Commonsense** | — | >85% | >90% | >92% | >95% |
| **Raspberry Pi 5 tok/s** | >100 | >50 | >50 | >50 | >40 |
| **Mac Mini M4 tok/s** | >100 | >120 | >120 | >120 | >100 |
| **BNC Efficiency** | 4× BPE | 4× BPE | 4× BPE | 4× BPE | 4× BPE |
| **Federated Devices** | — | — | — | 1,000+ | 10,000+ |

### 11.2 Ecosystem Metrics

| Metric | Phase 1 | Phase 3 | Phase 5 |
|--------|---------|---------|---------|
| **Hugging Face Downloads** | 10K | 100K | 1M+ |
| **GitHub Stars** | 1K | 5K | 20K+ |
| **Discord/Telegram Members** | 500 | 5,000 | 25,000+ |
| **Bharat Tiny Fellows** | 10 | 50 | 200+ |
| **Startup Partners** | 3 | 15 | 50+ |
| **Academic Citations** | 5 | 50 | 200+ |

### 11.3 Eulogik Brand Metrics

| Metric | Phase 1 | Phase 3 | Phase 5 |
|--------|---------|---------|---------|
| **YouTube Views (total)** | 100K | 1M | 10M+ |
| **Press Mentions** | 10 | 50 | 200+ |
| **Speaking Invitations** | 2 | 10 | 30+ |
| **"BRAHMI" as generic term** | Niche | Known in AI | "The BRAHMI of X" |
| **Eulogik inbound (startups)** | 5/month | 20/month | 50+/month |

---

## 12. The Eulogik Manifesto

> *"We did not come to build India's version of ChatGPT. We came to build something ChatGPT cannot be: an AI that reads Devanagari like a child learns to read, that blends Hinglish like a Delhi street vendor, that knows when to say 'Shubh Deepawali' and when to say 'Happy Diwali' — and that lives on the phone in your pocket, not in a data center in Virginia."*
>
> *"This is BRAHMI. This is Bharat-Tiny-LLM. This is Eulogik."*
>
> *"We are not building a model. We are building a mirror."*

---

*Document Version: 1.0*
*Eulogik Confidential — Build & GTM Plan*
