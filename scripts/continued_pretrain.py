"""
Minimal embed-warmup. Pure Python iterator (no DataLoader).
Trains only 200 new embedding + LM head rows via gradient-copying.
"""
import json, os, sys, math, time, random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def data_stream(path, tok, max_len=128, buf=2000):
    """Generator yielding token-ID chunks."""
    b = []
    for line in open(path):
        data = json.loads(line)
        for msg in data['messages']:
            ids = tok.encode(msg['content'])
            if len(ids) < 5: continue
            for i in range(0, len(ids), max_len):
                chunk = ids[i:i + max_len]
                if len(chunk) >= 5:
                    b.append(chunk)
                    if len(b) >= buf:
                        random.shuffle(b)
                        yield from b
                        b.clear()
    if b:
        random.shuffle(b)
        yield from b


class LearnableRows(nn.Module):
    def __init__(self, embed_rows, lmhead_rows):
        super().__init__()
        self.embed_rows = nn.Parameter(embed_rows)
        self.lmhead_rows = nn.Parameter(lmhead_rows)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/dev_expanded_test'
    corpus_path = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/train_gold_v3.jsonl'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '/Volumes/KIOXIA 1TB/continued_pretrain_v4'
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-4
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    print(f"Model: {model_path} | Steps: {steps} | LR: {lr}", flush=True)

    print("[1/6] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    our_tokens = {tid: t for tid, t in tok.added_tokens_decoder.items()
                  if not str(t).startswith('<') and not getattr(t, 'special', False)}
    new_ids = sorted(our_tokens.keys())
    n_new = len(new_ids)
    print(f"  → {n_new} new tokens (IDs {new_ids[0]}-{new_ids[-1]})")

    print("[2/6] Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.model.embed_tokens.weight.requires_grad_(True)
    model.lm_head.weight.requires_grad_(True)
    print(f"  → Loaded in {time.time()-t0:.0f}s. Vocab: {model.config.vocab_size}")

    embed = model.model.embed_tokens.weight
    lmhead = model.lm_head.weight
    hidden = model.config.hidden_size
    new_t = torch.tensor(new_ids, dtype=torch.long)

    print("[3/6] Setting up learner...")
    learner = LearnableRows(embed.data[new_ids].clone(), lmhead.data[new_ids].clone())
    optim = torch.optim.AdamW(learner.parameters(), lr=lr, weight_decay=0.01)
    print(f"  → {n_new * hidden * 2:,} trainable params")

    print("[4/6] Prepping data...")
    stream = data_stream(corpus_path, tok, max_len=128, buf=2000)

    # Collect initial val batches
    val_batches = []
    for i, chunk in enumerate(stream):
        if i >= 30: break
        if i >= 10:
            val_batches.append(torch.tensor([chunk], dtype=torch.long))

    print(f"  → {len(val_batches)} val batches collected")

    # Reset stream for training
    stream = data_stream(corpus_path, tok, max_len=128, buf=2000)

    print("[5/6] Initial validation...")
    model.eval()
    init_losses = []
    for vb in val_batches[:10]:
        with torch.no_grad():
            init_losses.append(model(vb, labels=vb).loss.item())
    init_loss = np.mean(init_losses)
    print(f"  → Init loss: {init_loss:.4f} (PPL: {math.exp(init_loss):.1f})")

    print("[6/6] Training...")
    model.train()
    step = 0
    best_loss = init_loss
    start_t = time.time()
    log_every = max(1, steps // 10)

    for chunk in stream:
        if step >= steps: break

        # Copy trained rows into model
        with torch.no_grad():
            embed.data[new_ids] = learner.embed_rows.data
            lmhead.data[new_ids] = learner.lmhead_rows.data

        batch = torch.tensor([chunk], dtype=torch.long)
        out = model(batch, labels=batch)
        loss = out.loss
        loss.backward()

        # Copy new-row gradients to learner
        with torch.no_grad():
            learner.embed_rows.grad = embed.grad[new_ids].clone()
            learner.lmhead_rows.grad = lmhead.grad[new_ids].clone()

        optim.step()
        optim.zero_grad()
        model.zero_grad()
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - start_t
            model.eval()
            vlosses = [model(vb, labels=vb).loss.item() for vb in val_batches[:5]]
            avg = np.mean(vlosses)
            impr = ((init_loss - avg) / init_loss * 100) if init_loss > 0 else 0
            print(f"  Step {step}/{steps} | Tr:{loss.item():.4f} | Val:{avg:.4f} "
                  f"(PPL:{math.exp(avg):.1f}) | {impr:+.1f}% | {step/elapsed:.2f} it/s")
            if avg < best_loss: best_loss = avg
            model.train()

    total_t = time.time() - start_t
    final_impr = ((init_loss - best_loss) / init_loss * 100) if init_loss > 0 else 0
    print(f"\nDone! {step} steps in {total_t:.0f}s ({step/total_t:.2f} it/s)")
    print(f"Init: {init_loss:.4f} → Best: {best_loss:.4f} ({final_impr:.1f}%)")

    print("Saving...")
    with torch.no_grad():
        embed.data[new_ids] = learner.embed_rows.data
        lmhead.data[new_ids] = learner.lmhead_rows.data
    model.save_pretrained(os.path.join(output_dir, 'model'))
    tok.save_pretrained(os.path.join(output_dir, 'tokenizer'))
    print(f"→ Saved to {output_dir}")


if __name__ == '__main__':
    main()
