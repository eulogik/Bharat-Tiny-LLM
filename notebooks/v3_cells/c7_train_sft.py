%%writefile train_sft.py
"""Brahmi v3 SFT — LoRA on 2xT4 via torchrun. Resume-safe, heartbeat, OOM-friendly."""
import argparse, json, os, random, time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, TrainerCallback)


class Heartbeat(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)

    def on_log(self, args, state, control, logs=None, **kw):
        if state.is_world_process_zero and logs and "loss" in logs:
            self.path.write_text(json.dumps(
                {"step": state.global_step, "loss": logs.get("loss"),
                 "ts": time.time()}))


def find_local_ckpt(out):
    cks = [p for p in Path(out).glob("checkpoint-*") if p.is_dir()]
    return max(cks, key=lambda p: int(p.name.split("-")[-1])) if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="sft_out")
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--warmup-rows", default="")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--resume-dir", default="")
    ap.add_argument("--heartbeat", default="progress.json")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)

    tok = AutoTokenizer.from_pretrained(args.tok_dir)
    if tok.chat_template is None:
        # Base tokenizers sometimes ship without a template — use ChatML (Qwen-style)
        tok.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
        )
        if rank == 0:
            print("base tokenizer had no chat template -> applied ChatML fallback")

    model = AutoModelForCausalLM.from_pretrained(
        os.environ["BASE_MODEL"], torch_dtype=torch.float16,
        attn_implementation="sdpa")

    # inject: resize to fit real token ids, then write mean-init + warmup rows
    d = load_file(args.delta)
    ids = d["ids"].tolist()
    assert ids == sorted(set(ids)) and len(ids) == int(ids[-1] - ids[0] + 1), \
        "delta ids must be unique & contiguous"
    model.resize_token_embeddings(ids[-1] + 1)
    emb = model.get_input_embeddings()
    with torch.no_grad():
        _dev, _dt = emb.weight.device, emb.weight.dtype
        emb.weight[ids] = d["rows"].to(device=_dev, dtype=_dt)
        if args.warmup_rows and Path(args.warmup_rows).exists():
            w = load_file(args.warmup_rows)
            emb.weight[w["ids"].tolist()] = w["rows"].to(device=_dev, dtype=_dt)
            if rank == 0:
                print("warmup rows applied")
    for p in emb.parameters():
        p.requires_grad_(False)   # embeddings stay frozen in SFT

    lcfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.config.use_cache = False
    if rank == 0:
        model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.data)["train"]

    def encode(ex):
        msgs = ex["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        fids = tok(full, add_special_tokens=False)["input_ids"][:args.seq]
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        labels = list(fids)
        cut = min(len(pids), len(fids))
        for i in range(cut):
            labels[i] = -100
        return {"input_ids": fids, "labels": labels}

    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=2,
                desc="tokenize")
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]),
                   num_proc=2)
    ds = ds.shuffle(seed=42)

    class Collator:
        def __call__(self, feats):
            maxlen = max(len(f["input_ids"]) for f in feats)
            pad_id = tok.pad_token_id or tok.eos_token_id or 0
            batch_ids, batch_labels, batch_mask = [], [], []
            for f in feats:
                n = len(f["input_ids"])
                padn = maxlen - n
                batch_ids.append(f["input_ids"] + [pad_id] * padn)
                batch_labels.append(f["labels"] + [-100] * padn)
                batch_mask.append([1] * n + [0] * padn)
            t = lambda x: torch.tensor(x, dtype=torch.long)
            return {"input_ids": t(batch_ids),
                    "labels": t(batch_labels),
                    "attention_mask": t(batch_mask)}

    resume = args.resume_dir or ""
    local_ck = find_local_ckpt(args.out)
    if not resume and local_ck:
        resume = str(local_ck)
    if rank == 0 and resume:
        print("resuming from:", resume)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        bf16=False,                      # T4 has no bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        dataloader_num_workers=2,
        seed=42,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(),
                      callbacks=[Heartbeat(args.heartbeat)])
    trainer.train(resume_from_checkpoint=resume or None)

    if rank == 0:
        final = Path(args.out) / "final_adapter"
        trainer.model.save_pretrained(final)
        tok.save_pretrained(Path(args.out) / "tokenizer")
        (Path(args.out) / "DONE").write_text("ok")
        print("SFT complete ->", final)


if __name__ == "__main__":
    import traceback, sys
    _err_log = Path(os.environ.get("TRAIN_ERROR_LOG", "train_error.log"))
    try:
        main()
    except Exception:
        with open(_err_log, "w") as _f:
            traceback.print_exc(file=_f)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
