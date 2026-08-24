"""
# Bharat-Tiny-LLM v2: Continued Pre-training for Expanded Vocabulary

Run this in Google Colab with a T4 GPU (free tier).

This script warms up the 300 new Devanagari embedding rows injected
into Qwen2.5-1.5B. Without this step, the new tokens compress text
but the model doesn't know their meanings yet.

## Usage
1. Open https://colab.research.google.com
2. Runtime → Change runtime type → T4 GPU
3. Copy this entire cell and run
"""

# %% [markdown]
# ## 1. Install dependencies

# %%
import subprocess, sys, os, json
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "transformers", "torch", "datasets", "accelerate", "huggingface_hub"])

# %% [markdown]
# ## 2. Load expanded model

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_ID = "eulogik/Bharat-Tiny-LLM-v2"
print(f"Loading {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

# Identify new token IDs
new_ids = sorted(
    tid for tid, t in tok.added_tokens_decoder.items()
    if not str(t).startswith('<') and not getattr(t, 'special', False)
)
print(f"Found {len(new_ids)} new tokens (IDs {new_ids[0]}–{new_ids[-1]})")

# %% [markdown]
# ## 3. Freeze backbone, only train new embeddings

# %%
# Freeze everything
for p in model.parameters():
    p.requires_grad = False

# Unfreeze only new embedding + LM head rows
model.model.embed_tokens.weight.requires_grad_(True)
model.lm_head.weight.requires_grad_(True)

class LearnableRows(torch.nn.Module):
    def __init__(self, embed, lmhead):
        super().__init__()
        self.embed = torch.nn.Parameter(embed)
        self.lmhead = torch.nn.Parameter(lmhead)

hidden = model.config.hidden_size
embed_w = model.model.embed_tokens.weight
lmhead_w = model.lm_head.weight

learner = LearnableRows(
    embed_w.data[new_ids].clone(),
    lmhead_w.data[new_ids].clone()
)
optim = torch.optim.AdamW(learner.parameters(), lr=5e-4, weight_decay=0.01)

print(f"Training {len(new_ids) * hidden * 2:,} params "
      f"(out of {sum(p.numel() for p in model.parameters()):,})")

# %% [markdown]
# ## 4. Prepare data
# We use our gold corpus uploaded to HF. If not available,
# we fall back to a sample of Hindi Wikipedia.

# %%
import numpy as np

try:
    dataset = load_dataset("text", data_files="train_gold_v3.jsonl", split="train")
except:
    print("Local corpus not found, using Hindi Wikipedia sample")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    dataset = dataset.take(5000)

def tokenize_fn(examples):
    texts = examples.get("text", examples.get("content", [""]))
    if isinstance(texts, str):
        texts = [texts]
    ids_list = []
    for t in texts:
        # Extract from message format if needed
        if isinstance(t, str) and t.startswith("{"):
            try:
                data = json.loads(t)
                t = " ".join(m["content"] for m in data.get("messages", []))
            except:
                pass
        ids = tok.encode(t)
        for i in range(0, len(ids), 256):
            chunk = ids[i:i+256]
            if len(chunk) >= 10:
                ids_list.append(chunk)
    return {"input_ids": ids_list}

# %% [markdown]
# ## 5. Training loop

# %%
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import math, time

def collate(batch):
    ids = [torch.tensor(ex["input_ids"][0], dtype=torch.long)
           for ex in batch if len(ex.get("input_ids", [])) > 0]
    # Handle list-of-dicts format
    if not ids:
        ids = [torch.tensor(ex["input_ids"] if isinstance(ex["input_ids"], list) else [ex["input_ids"]], dtype=torch.long)
               for ex in batch]
    if not ids:
        return torch.zeros(1, 1, dtype=torch.long)
    mx = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), mx, dtype=torch.long)
    for i, x in enumerate(ids):
        pad[i, :len(x)] = x
    return pad

# Prepare data
all_chunks = []
for i, ex in enumerate(dataset):
    if i >= 2000: break
    text = ex.get("text", ex.get("content", ""))
    if isinstance(text, str) and text.startswith("{"):
        try:
            data = json.loads(text)
            text = " ".join(m["content"] for m in data.get("messages", []))
        except:
            pass
    ids = tok.encode(text)
    for j in range(0, len(ids), 256):
        chunk = ids[j:j+256]
        if len(chunk) >= 10:
            all_chunks.append(chunk)
    if i == 20:
        print(f"Sample chunk: {len(all_chunks[0])} tokens")
        print(f"  Contains new tokens: {any(tid in new_ids for tid in all_chunks[0])}")

print(f"Total chunks: {len(all_chunks)}")
all_chunks = all_chunks[:2000]

loader = DataLoader(all_chunks, batch_size=2, shuffle=True, collate_fn=collate)

# Val set
val_chunks = all_chunks[-100:]
val_loader = DataLoader(val_chunks, batch_size=2, collate_fn=collate)

NUM_STEPS = 500
optim = torch.optim.AdamW(learner.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optim, int(NUM_STEPS*0.1), NUM_STEPS)

# Initial validation
model.eval()
init_losses = []
for batch in val_loader:
    with torch.no_grad():
        init_losses.append(model(batch, labels=batch).loss.item())
init_loss = np.mean(init_losses)
print(f"\nInitial loss: {init_loss:.4f} (PPL: {math.exp(init_loss):.1f})")

# Train
model.train()
step = 0
best_loss = init_loss
start_t = time.time()

for epoch in range(5):
    for batch in loader:
        if step >= NUM_STEPS: break

        # Copy learner rows to model
        with torch.no_grad():
            embed_w.data[new_ids] = learner.embed.data
            lmhead_w.data[new_ids] = learner.lmhead.data

        out = model(batch, labels=batch)
        loss = out.loss
        loss.backward()

        with torch.no_grad():
            learner.embed.grad = embed_w.grad[new_ids].clone()
            learner.lmhead.grad = lmhead_w.grad[new_ids].clone()

        optim.step()
        optim.zero_grad()
        model.zero_grad()
        step += 1

        if step % 50 == 0:
            model.eval()
            v = np.mean([model(vb, labels=vb).loss.item()
                        for vb in [next(iter(val_loader))]])
            impr = ((init_loss - v) / init_loss * 100)
            elapsed = time.time() - start_t
            print(f"Step {step}/{NUM_STEPS} | Tr:{loss.item():.4f} | "
                  f"Val:{v:.4f} (PPL:{math.exp(v):.1f}) | {impr:+.1f}% | "
                  f"{step/elapsed:.2f} it/s")
            if v < best_loss:
                best_loss = v
            model.train()

elapsed = time.time() - start_t
print(f"\nDone! {step} steps in {elapsed:.0f}s")
print(f"Init loss: {init_loss:.4f} → Best: {best_loss:.4f} "
      f"({(init_loss-best_loss)/init_loss*100:.1f}% improvement)")

# %% [markdown]
# ## 6. Save and upload

# %%
# Copy final embeddings to model
with torch.no_grad():
    embed_w.data[new_ids] = learner.embed.data
    lmhead_w.data[new_ids] = learner.lmhead.data

# Save locally
model.save_pretrained("bharat-tiny-llm-v2-trained")
tok.save_pretrained("bharat-tiny-llm-v2-trained")
print("Model saved locally")

# Upload to HF
from huggingface_hub import HfApi, create_repo, login
login(token="YOUR_HF_TOKEN")  # Replace with your token or use env var

api = HfApi()
api.upload_folder(
    folder_path="bharat-tiny-llm-v2-trained",
    repo_id="eulogik/Bharat-Tiny-LLM-v2",
    repo_type="model",
    commit_message="Continued pre-training: warmed up new Devanagari embeddings",
)

print("✅ Uploaded to HF!")
print(f"https://huggingface.co/eulogik/Bharat-Tiny-LLM-v2")
