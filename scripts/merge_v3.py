"""Merge Qwen3-1.7B + token injection + warmup rows + SFT LoRA → standalone model."""
import torch, json, os, sys
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGE_DIR = Path("/tmp/v3_merge")
BASE = "Qwen/Qwen3-1.7B-Base"
OUT = Path("/Volumes/KIOXIA 1TB/brahmi-v3-merged")

# 1. Load base model
print("loading base model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)

# 2. Resize embeddings for 300 new tokens
d = load_file(str(MERGE_DIR / "embed_delta.safetensors"))
ids = d["ids"].tolist()
first_id = ids[0]
model.resize_token_embeddings(ids[-1] + 1)
emb = model.get_input_embeddings()
print(f"resized to {emb.weight.shape[0]} tokens, first new id: {first_id}")

# 3. Apply mean-init delta
with torch.no_grad():
    emb.weight[ids] = d["rows"].to(device=emb.weight.device, dtype=emb.weight.dtype)

# 4. Apply warmup rows (override mean-init with trained embeddings)
w = load_file(str(MERGE_DIR / "warmup_rows.safetensors"))
with torch.no_grad():
    emb.weight[w["ids"].tolist()] = w["rows"].to(device=emb.weight.device, dtype=emb.weight.dtype)
print(f"warmup rows applied ({len(w['ids'])} rows)")

# 5. Load LoRA adapter
print("loading LoRA adapter...", flush=True)
from peft import PeftModel
model = PeftModel.from_pretrained(model, str(MERGE_DIR / "sft" / "final"))
print("LoRA loaded, merging...")
model = model.merge_and_unload()
print("merged")

# 6. Save merged model
OUT.mkdir(parents=True, exist_ok=True)
print(f"saving to {OUT}...", flush=True)
model.save_pretrained(str(OUT))

# 7. Save tokenizer
tok = AutoTokenizer.from_pretrained(str(MERGE_DIR / "sft" / "final_tokenizer"))
tok.save_pretrained(str(OUT / "tokenizer"))

print(f"done — model saved to {OUT}")
print(f"  params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
