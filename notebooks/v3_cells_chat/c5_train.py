#@title 4 · Train conversational SFT {display-mode:"form"}
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from pathlib import Path

class Heartbeat(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if state.is_world_process_zero and logs and "loss" in logs:
            (WORK / "chat_progress.json").write_text(json.dumps(
                {"step": state.global_step, "loss": logs.get("loss"),
                 "eval_loss": logs.get("eval_loss"),
                 "ts": time.time()}))

def find_local_ckpt(out):
    cks = [p for p in Path(out).glob("checkpoint-*") if p.is_dir()]
    return max(cks, key=lambda p: int(p.name.split("-")[-1])) if cks else None

if not STATE.get("train_done"):
    t0 = time.time()

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

    targs = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        dataloader_num_workers=2,
        seed=42,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=Collator(),
        callbacks=[Heartbeat()],
    )

    resume_dir = find_local_ckpt(str(OUT_DIR))
    print(f"training | epochs={EPOCHS} eff_bs={BS*GRAD_ACCUM} "
          f"| resume={resume_dir or 'none'}")
    trainer.train(resume_from_checkpoint=str(resume_dir) if resume_dir else None)

    final = OUT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(final))
    tok.save_pretrained(str(OUT_DIR / "tokenizer"))
    (OUT_DIR / "DONE").write_text("ok")

    elapsed = time.time() - t0
    STATE["train_done"] = True
    STATE["train_time_min"] = round(elapsed / 60, 1)
    save_state(STATE)
    print(f"train done in {elapsed/60:.1f} min → {final}")
else:
    print("training already done")
