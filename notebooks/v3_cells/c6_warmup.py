#@title 5 · PHASE 3 — Warmup: train ONLY the 300 new embedding rows (~40 min GPU) {display-mode:"form"}
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file

WARMUP_PATH = WORK / "warmup_rows.safetensors"


class SplitEmbedding(nn.Module):
    """Frozen original embedding + trainable block of `budget` new rows."""

    def __init__(self, orig_emb, rows, new_ids):
        super().__init__()
        self.orig = orig_emb
        for p in self.orig.parameters():
            p.requires_grad_(False)
        self.new = nn.Parameter(rows.clone().float())
        self.register_buffer("new_ids", new_ids, persistent=False)
        self.first_id = int(new_ids.min())

    def forward(self, ids):
        out = self.orig(ids)
        mask = ids >= self.first_id
        if mask.any():
            extra = torch.zeros_like(out)
            hit_ids = ids[mask] - self.first_id
            extra[mask] = self.new[hit_ids].to(out.dtype)
            return out + extra
        return out


def build_model_with_split(tok):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    base_vocab = model.get_input_embeddings().num_embeddings
    d = load_file(str(DELTA_PATH))
    assert d["ids"].tolist() == list(range(base_vocab, base_vocab + BUDGET)), \
        "delta ids must be contiguous and match vocab layout"
    model.resize_token_embeddings(base_vocab + BUDGET)
    split = SplitEmbedding(model.get_input_embeddings(),
                           d["rows"].float(), d["ids"])
    model.set_input_embeddings(split)
    if WARMUP_PATH.exists():
        w = load_file(str(WARMUP_PATH))
        with torch.no_grad():
            split.new.copy_(w["rows"].float())
        print("resumed warmup rows from artifact")
    return model, split


if not STATE.get("phase3"):
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    model, split = build_model_with_split(tok)
    model.cuda()

    corpus = get_corpus_handle()
    new_set = set(d0["ids"].tolist()) if (d0 := load_file(str(DELTA_PATH))) else None
    chunks, buf_ok = [], True
    for line in corpus:
        if len(chunks) >= 20000:
            break
        try:
            text = " ".join(m["content"] for m in json.loads(line)["messages"])[:4000]
        except Exception:
            continue
        ids = tok.encode(text)
        for i in range(0, len(ids) - WARMUP_SEQ, WARMUP_SEQ):
            ch = ids[i:i + WARMUP_SEQ + 1]
            if any(t in new_set for t in ch[:-1]):
                chunks.append(ch)
            if len(chunks) >= 20000:
                break
    corpus.close()
    print(f"warmup chunks: {len(chunks)}")

    opt = torch.optim.AdamW([split.new], lr=WARMUP_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_MAX=WARMUP_STEPS)
    dl = DataLoader(chunks, batch_size=WARMUP_BS, shuffle=True, drop_last=True)

    start_step = STATE.get("warmup_step", 0)
    losses, step = [], start_step
    model.train()
    t0 = time.time()
    while step < WARMUP_STEPS:
        for batch in dl:
            x = torch.tensor(batch)[:, :-1].cuda(non_blocking=True)
            y = torch.tensor(batch)[:, 1:].cuda(non_blocking=True)
            logits = model(input_ids=x).logits
            loss = nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                recent = sum(losses[-200:]) / min(200, len(losses))
                rate = (step - start_step) / max(time.time() - t0, 1)
                print(f"step {step}/{WARMUP_STEPS} loss {recent:.3f} ({rate:.2f} it/s)",
                      flush=True)
            if step % 1000 == 0 or step == WARMUP_STEPS:
                save_file({"ids": split.new_ids.cpu(),
                           "rows": split.new.detach().to(torch.float16).cpu()},
                          str(WARMUP_PATH))
                upload_file(str(WARMUP_PATH), "warmup_rows.safetensors",
                            ARTIFACTS_REPO, repo_type="model", token=HF_TOKEN)
                STATE["warmup_step"] = step
                save_state(STATE)
            if step >= WARMUP_STEPS:
                break

    final = sum(losses[-100:]) / max(1, min(100, len(losses)))
    STATE["phase3"] = True
    STATE["warmup_final_loss"] = round(final, 4)
    save_state(STATE)
else:
    print("phase3 done | final warmup loss:", STATE.get("warmup_final_loss"))
