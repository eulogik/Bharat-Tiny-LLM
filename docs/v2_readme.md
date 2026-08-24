# Bharat-Tiny-LLM v2: Surgical Brahmic Vocabulary Retrofit for Qwen2.5

## What

A **300-token Devanagari vocabulary injection** into Qwen2.5-1.5B's byte-level BPE tokenizer, achieving **33.8% token reduction** on Hindi text with zero architecture changes. The first application of the BrahmicTokenizer-131K surgical retrofit methodology to the Qwen architecture.

## Why It Matters

Qwen2.5's tokenizer had **zero Devanagari-specific tokens** — Hindi text was tokenized as raw UTF-8 bytes at ~1.03 tokens/character (worse than 1:1). This is not a training issue but a structural vocab gap. By surgically injecting 300 high-frequency Devanagari subwords, we compress Hindi text by 33.8%, giving edge users 33.8% more effective context length.

## Results

| Metric | Baseline | Improved | Gain |
|--------|----------|----------|------|
| Chars/token (Devanagari) | 0.96 | 1.51 | **+57%** |
| Token savings | — | — | **33.8%** |
| Effective context gain | 32K | ~43K | **+34%** |
| Model size (Q4 MLX) | 880 MB | 880 MB | **Same** |
| New tokens added | 0 | 300 | **+300** |

## Methodology

1. **Audit**: Counted 0 Devanagari tokens in Qwen2.5's 151K vocab
2. **Analyze**: Sampled 10K corpus lines → identified best Devanagari n-grams by token-savings × frequency
3. **Inject**: Used HuggingFace `tokenizer.add_tokens()` — no BPE merge learning needed
4. **Initialize**: New embeddings = mean of constituent byte-token embeddings (gradient-copying training pattern)
5. **Quantize**: MLX Q4 (4.5 bpw, affine, group_size=64)

## Models

| Model | Format | Size | Link |
|-------|--------|------|------|
| v2 Expanded | PyTorch fp16 | 3.6 GB | [HF](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2) |
| v2 MLX Q4 | MLX q4 | 880 MB | [HF](https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2-MLX) |

## Usage

```python
# Load from HuggingFace
from mlx_lm import load, generate
model, tokenizer = load("eulogik/Bharat-Tiny-LLM-v2-MLX")

# 58% fewer tokens for Hindi
test = "ज़रूरी बात है क्या करते हो"
ids = tokenizer.encode(test)  # 11 vs 26 tokens!
response = generate(model, tokenizer, test, max_tokens=100)
```

## Continued Pre-Training

The new embeddings are initialized as mean of constituent byte tokens. For best generation quality, run [continued_pretrain_colab.py](scripts/continued_pretrain_colab.py) on a GPU (~1 hour on T4).

## Token List

300 injected Devanagari subwords (common akshara clusters, verb forms, postpositions):
- Verb stems: कर, हो, सक, ले, दे, रख, आ, जा, पड़
- Common words: क्या, नहीं, है, हैं, में, का, की, से, को, पर
- Compounds: ज़रूरी, कैसे, जैसे, तुम्हारे, अपने, बहुत
- Matras and modifiers: ैं, ्या, ी, े, ो

## Citation

```bibtex
@misc{brahmi2025vocab,
  title = {Surgical Brahmic Vocabulary Retrofit for Qwen2.5},
  author = {Gautam Kishore},
  year = {2025},
  url = {https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2}
}
```
