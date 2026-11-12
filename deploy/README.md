# Bharat-Tiny-LLM v4 — local deployment

The v4 artifact is a **system**: stock Qwen3.5-2B Q4_K_M (~1.2 GB) plus a local
router, a retrieval-only health layer, and post-generation guards. An
experimental RAG index is included but is **off by default**. Everything below
is offline after the model file and this repository are on disk.

> **Do not call the raw model for health questions.** Use `scripts/rag_ask.py`
> or the same router/safety layer in your app. Health answers are fixed,
> vetted protocols; the model never generates medical advice.

## 1. Recommended entry point

```bash
# start the model server
llama-server \
  -m Qwen3.5-2B-Q4_K_M.gguf \
  -c 4096 -np 1 \
  --host 127.0.0.1 --port 8096 &

# ask through the router + health layer (RAG off by default)
venv/bin/python scripts/rag_ask.py \
  --server http://127.0.0.1:8096/v1 \
  --model Qwen3.5-2B-Q4_K_M.gguf \
  --query "PM Kisan ka status kaise check karein?"
```

`rag_ask.py` does:

1. domain routing (`scripts/domain_prompt.py`);
2. for health, a fixed protocol or a vetted generic referral — **no model
   call** (`scripts/safety_protocol.py`);
3. for everything else, the base model plus the crop/animal and loop guards;
4. prints the routed domain and latency. Add `--rag` to enable the
   experimental BM25 retrieval, which also prints its sources.

**Retrieval caveat.** RAG is **off by default** because it is not reliable
enough yet: a “CSC centre” question retrieves a tomato advisory, and crop
chunks are not crop-specific. If you enable `--rag`, verify the retrieved
sources before trusting the answer.

## 2. Raw llama.cpp server (non-health only)

```bash
llama-server \
  -m Qwen3.5-2B-Q4_K_M.gguf \
  -c 4096 -np 1 \
  --host 127.0.0.1 --port 8096
```

```bash
curl -s http://127.0.0.1:8096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role":"system","content":"You are Bharat-Tiny-LLM. Reply in the user'"'"'s language and script. Never invent live prices, dates or scheme details."},
      {"role":"user","content":"Bhai, PM Kisan ka status kaise check karein?"}
    ],
    "temperature": 0.2,
    "max_tokens": 300,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**Always send `enable_thinking: false`.** Qwen3.5 is a reasoning model; without
this flag it can spend the whole token budget inside a think block and never
answer.

## 3. Ollama

```bash
ollama create bharat-v4 -f deploy/Modelfile.bharat-v4
ollama run bharat-v4
```

Edit `FROM` in the Modelfile to the absolute path of the GGUF. The Modelfile is
for **non-health** chat; use the Python entry point for health.

## 4. Android / phone

The smallest path is llama.cpp's Android build (`llama-server` or the
`llama.android` sample) plus the Q4_K_M GGUF. Target devices need ~2 GB free
RAM. The model does not require Google Play Services or a network connection.

For a ₹8,000-class phone, test with:
- Q4_K_M (1.2 GB) first; if RAM is tight, Q3_K_M or IQ3_XXS
- context 2,048 for the first release (voice turns are short)
- `enable_thinking=false`, `temperature=0.2`, `repeat_penalty=1.1`

## 5. TTS

macOS offline:

```bash
say -v Lekha "नमस्ते, मैं आपकी मदद कर सकता हूँ।"
```

For Linux/Android, bundle a Piper Hindi voice (e.g. `hi_IN-pratham-medium`)
and call it from the voice loop. Do not claim free-form ASR: see
`docs/asr_decision.md`.
