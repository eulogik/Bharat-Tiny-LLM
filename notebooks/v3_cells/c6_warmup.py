#@title 5 · PHASE 3 — Warmup: train ONLY the 300 new embedding rows (~40 min GPU) {display-mode:"form"}
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file

WARMUP_PATH = WORK / "warmup_rows.safetensors"

# resume mid-warmup: pull partial artifact saved by a previous session
if not WARMUP_PATH.exists() and STATE.get("warmup_step"):
    try:
        hf_hub_download(repo_id=ARTIFACTS_REPO, filename="warmup_rows.safetensors",
                        repo_type="model", token=HF_TOKEN, local_dir=str(WORK),
                        force_download=True)
        print(f"resuming warmup from step {STATE['warmup_step']}")
    except Exception as e:
        print("no remote warmup artifact yet:", type(e).__name__)

if not STATE.get("phase3"):
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                                 torch_dtype=torch.float16).cuda()

    d = load_file(str(DELTA_PATH))
    ids = d["ids"].tolist()
    assert ids == sorted(set(ids)) and len(ids) == BUDGET \
        and ids[-1] - ids[0] == BUDGET - 1, "delta ids must be unique & contiguous"
    first_id = ids[0]

    # NOTE: Qwen3-1.7B TIES embed_tokens & lm_head — resize keeps them tied,
    # so writing rows here makes new tokens both input-embeddable AND predictable.
    model.resize_token_embeddings(ids[-1] + 1)
    emb = model.get_input_embeddings()
    dev, dt = emb.weight.device, emb.weight.dtype
    with torch.no_grad():
        emb.weight[ids] = d["rows"].to(device=dev, dtype=dt)
        if WARMUP_PATH.exists():
            w = load_file(str(WARMUP_PATH))
            emb.weight[w["ids"].tolist()] = w["rows"].to(device=dev, dtype=dt)
            print("resumed warmup rows from artifact")

    for p in model.parameters():
        p.requires_grad_(False)
    emb.weight.requires_grad_(True)

    corpus = get_corpus_handle()
    id_set = set(ids)
    chunks = []
    for line in corpus:
        if len(chunks) >= 20000:
            break
        try:
            text = " ".join(m["content"] for m in json.loads(line)["messages"])[:4000]
        except Exception:
            continue
        tks = tok.encode(text)
        for i in range(0, len(tks) - WARMUP_SEQ, WARMUP_SEQ):
            ch = tks[i:i + WARMUP_SEQ + 1]
            if any(t in id_set for t in ch[:-1]):
                chunks.append(ch)
            if len(chunks) >= 20000:
                break
    corpus.close()
    print(f"warmup chunks: {len(chunks)}")

    opt = torch.optim.AdamW([emb.weight], lr=WARMUP_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=WARMUP_STEPS)
    def _collate(items):
        return torch.tensor(items, dtype=torch.long)

    dl = DataLoader(chunks, batch_size=WARMUP_BS, shuffle=True,
                    drop_last=True, collate_fn=_collate)

    start_step = STATE.get("warmup_step", 0)
    losses, step = [], start_step
    model.train()
    t0 = time.time()
    while step < WARMUP_STEPS:
        oom_streak = 0
        for batch in dl:
            x = batch[:, :-1].cuda(non_blocking=True)
            y = batch[:, 1:].cuda(non_blocking=True)
            try:
                logits = model(input_ids=x).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]).float(), y.view(-1))
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_streak += 1
                if oom_streak > 25:
                    raise RuntimeError("warmup OOM: >25 consecutive skipped batches")
                continue
            oom_streak = 0
            # mask grads: only the 300 injected rows may move
            g = emb.weight.grad
            if g is not None:
                g[:first_id] = 0
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
                with torch.no_grad():
                    rows = emb.weight[ids].detach().to(torch.float16).cpu()
                save_file({"ids": d["ids"], "rows": rows}, str(WARMUP_PATH))
                upload_file(path_or_fileobj=str(WARMUP_PATH),
                            path_in_repo="warmup_rows.safetensors",
                            repo_id=ARTIFACTS_REPO, repo_type="model",
                            token=HF_TOKEN)
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
