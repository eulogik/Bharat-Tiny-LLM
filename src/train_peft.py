"""PEFT/Transformers LoRA training on MPS (Metal)."""

import os, sys, json, math, time, gc
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
TRAIN_DATA = "data/processed/train.jsonl"
VAL_DATA = "data/processed/valid.jsonl"
OUTPUT_DIR = "models/adapters/qwen_peft_v1"

# LoRA (matches MLX config: 16 layers, r=8, alpha=16)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAYERS_TO_TRANSFORM = list(range(12, 28))

# Training
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 512
LR = 5e-5
WARMUP_STEPS = 200
TOTAL_STEPS = 100_000
EVAL_STEPS = 500
SAVE_STEPS = 5000
LOGGING_STEPS = 50
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
SCHEDULER = "cosine"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16

print(f"Device: {DEVICE}, dtype: {DTYPE}")
print(f"Transformers: {__import__('transformers').__version__}, PEFT: {__import__('peft').__version__}")


# ── Dataset ─────────────────────────────────────────────────────────────
class ChatMLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                if "messages" in item:
                    self.data.append(item["messages"])
        print(f"Loaded {len(self.data)} examples from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]
        # Build full ChatML text and track which spans are assistant responses
        text = ""
        assistant_spans = []  # (start_char, end_char) in the final text
        for m in messages:
            role = m["role"]
            content = m["content"]
            start = len(text)
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            end = len(text)
            if role == "assistant":
                assistant_spans.append((start, end))
        # Add trailing assistant prompt for generation (no loss computed here)
        text += "<|im_start|>assistant\n"

        # Tokenize
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]

        # Build labels: -100 for non-assistant tokens
        labels = torch.full_like(input_ids, -100)
        for start_char, end_char in assistant_spans:
            # Map character spans to token spans
            for tok_idx in range(len(input_ids)):
                tok_start = tokens.token_to_chars(tok_idx).start if tokens.token_to_chars(tok_idx) else None
                tok_end = tokens.token_to_chars(tok_idx).end if tokens.token_to_chars(tok_idx) else None
                if tok_start is not None and tok_start >= start_char and tok_end <= end_char:
                    labels[tok_idx] = input_ids[tok_idx]

        # If truncation happened, the end of the conversation might be cut off
        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"][0],
            "labels": labels,
        }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        use_cache=False,
    )
    model = model.to(DEVICE)
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        layers_to_transform=LAYERS_TO_TRANSFORM,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = ChatMLDataset(TRAIN_DATA, tokenizer, MAX_SEQ_LEN)
    val_dataset = ChatMLDataset(VAL_DATA, tokenizer, MAX_SEQ_LEN)

    # Collator
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        max_steps=TOTAL_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=SCHEDULER,
        gradient_checkpointing=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        run_name="qwen_peft_v1",
        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    # Save training metrics
    with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f"Training complete. Model saved to {OUTPUT_DIR}/final")
    print(f"Metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
