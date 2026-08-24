"""
Train ONLY the 200 new embedding/LMHead rows. All other params frozen.
Writes progress to log file. Uses CPU, pure Python generator for data.
"""
import json, os, sys, math, time, random, logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

log_path = sys.argv[3] if len(sys.argv) > 3 else '/Volumes/KIOXIA 1TB/continued_pretrain_v4'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, 'train.log'),
    level=logging.INFO, format='%(asctime)s %(message)s',
    force=True
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/dev_expanded_test'
corpus = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/train_gold_v3.jsonl'
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-4
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
device = torch.device('cpu')

log.info(f"Train embeds: {model_path} | {steps} steps | LR {lr}")

tok = AutoTokenizer.from_pretrained(model_path)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

our = {tid: t for tid, t in tok.added_tokens_decoder.items()
       if not str(t).startswith('<') and not getattr(t, 'special', False)}
new_ids = sorted(our.keys())
n_new = len(new_ids)
log.info(f"New tokens: {n_new} (IDs {new_ids[0]}-{new_ids[-1]})")

log.info("Loading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
for p in model.parameters():
    p.requires_grad = False
model.model.embed_tokens.weight.requires_grad_(True)
model.lm_head.weight.requires_grad_(True)
log.info(f"Loaded in {time.time()-t0:.0f}s. Vocab: {model.config.vocab_size}")

embed = model.model.embed_tokens.weight
lmhead = model.lm_head.weight
hidden = model.config.hidden_size

class Learner(nn.Module):
    def __init__(self, e, l):
        super().__init__()
        self.e = nn.Parameter(e); self.l = nn.Parameter(l)

learner = Learner(embed.data[new_ids].clone(), lmhead.data[new_ids].clone())
optim = torch.optim.AdamW(learner.parameters(), lr=lr, weight_decay=0.01)
log.info(f"Params: {n_new * hidden * 2:,}")

def stream():
    buf = []
    for line in open(corpus):
        data = json.loads(line)
        for msg in data['messages']:
            ids = tok.encode(msg['content'])
            if len(ids) < 5: continue
            for i in range(0, len(ids), 128):
                chunk = ids[i:i+128]
                if len(chunk) >= 5:
                    buf.append(chunk)
                    if len(buf) >= 2000:
                        random.shuffle(buf)
                        yield from buf
                        buf.clear()
    if buf: random.shuffle(buf); yield from buf

log.info("Collecting val batches...")
gen = stream()
val_batches = []
for i, c in enumerate(gen):
    if i >= 20: break
    if i >= 5: val_batches.append(torch.tensor([c], dtype=torch.long))
log.info(f"{len(val_batches)} val batches")

log.info("Estimating initial loss...")
model.eval()
init_losses = []
for vb in val_batches:
    with torch.no_grad():
        init_losses.append(model(vb, labels=vb).loss.item())
init_loss = np.mean(init_losses)
log.info(f"Init loss: {init_loss:.4f} PPL: {math.exp(init_loss):.1f}")

gen = stream()
model.train()
step = 0
best_loss = init_loss
start_t = time.time()
log_every = max(1, steps // 10)

for chunk in gen:
    if step >= steps: break
    with torch.no_grad():
        embed.data[new_ids] = learner.e.data
        lmhead.data[new_ids] = learner.l.data
    batch = torch.tensor([chunk], dtype=torch.long)
    out = model(batch, labels=batch)
    loss = out.loss
    loss.backward()
    with torch.no_grad():
        learner.e.grad = embed.grad[new_ids].clone()
        learner.l.grad = lmhead.grad[new_ids].clone()
    optim.step()
    optim.zero_grad()
    model.zero_grad()
    step += 1
    if step % log_every == 0:
        model.eval()
        v = np.mean([model(vb, labels=vb).loss.item() for vb in val_batches])
        impr = ((init_loss - v)/init_loss*100) if init_loss > 0 else 0
        log.info(f"Step {step}/{steps} | Tr:{loss.item():.4f} | Val:{v:.4f} PPL:{math.exp(v):.1f} | {impr:+.1f}% | {step/(time.time()-start_t):.2f} it/s")
        if v < best_loss: best_loss = v
        model.train()

elapsed = time.time() - start_t
final = ((init_loss - best_loss)/init_loss*100) if init_loss > 0 else 0
log.info(f"Done {step} steps in {elapsed:.0f}s ({step/elapsed:.2f} it/s)")
log.info(f"Init: {init_loss:.4f} -> Best: {best_loss:.4f} ({final:.1f}%)")

with torch.no_grad():
    embed.data[new_ids] = learner.e.data
    lmhead.data[new_ids] = learner.l.data
model.save_pretrained(os.path.join(log_path, 'model'))
tok.save_pretrained(os.path.join(log_path, 'tokenizer'))
log.info(f"Saved to {log_path}")
log.info("DONE")
