#@title 3 · Load & tokenize chat dataset {display-mode:"form"}
from datasets import load_dataset

if not STATE.get("tok_done"):
    t0 = time.time()
    print(f"loading {CHAT_DATA} ...", flush=True)
    ds = load_dataset(CHAT_DATA, split="train", token=HF_TOKEN)
    print(f"  raw rows: {len(ds)}")

    # sample down to 20K if needed (dataset is exactly 20K but safety check)
    if len(ds) > 20000:
        ds = ds.shuffle(seed=42).select(range(20000))

    def tokenize_chat(ex):
        msgs = ex["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # Qwen3 chat template (native)
        full = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        fids = tok(full, add_special_tokens=False)["input_ids"][:MAX_SEQ]
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        labels = list(fids)
        cut = min(len(pids), len(fids))
        for i in range(cut):
            labels[i] = -100
        return {"input_ids": fids, "labels": labels}

    ds = ds.map(tokenize_chat, remove_columns=ds.column_names,
                num_proc=2, desc="tokenize")
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]),
                   num_proc=2)
    ds = ds.shuffle(seed=42)

    # 95/5 train/eval split
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

    print(f"  train: {len(train_ds)} | eval: {len(eval_ds)} | done in {time.time()-t0:.0f}s")
    STATE["tok_done"] = True
    STATE["train_rows"] = len(train_ds)
    STATE["eval_rows"] = len(eval_ds)
    save_state(STATE)
else:
    print("tokenization already done")
