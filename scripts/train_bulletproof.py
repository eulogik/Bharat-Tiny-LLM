#!/usr/bin/env python3
"""Bulletproof training wrapper for Bharat-Tiny-LLM."""

import os, sys, json, signal, time, subprocess
from pathlib import Path
from datetime import datetime

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-bf16"
ADAPTER_DIR = Path("./models/adapters/qwen_v2")
DATA_DIR = Path("./data/processed")
LOG_FILE = Path("./training_v2.log")
METRICS_FILE = Path("./training_metrics.jsonl")
TOTAL_ITERS = 110000
SAVE_EVERY = 5000
EVAL_EVERY = 500
MAX_SEQ_LENGTH = 2048
LORA_LAYERS = 16
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

best_val_loss = float("inf")
best_iter = 0
metrics = []
training_running = True

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_metrics():
    with open(METRICS_FILE, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

def find_last_checkpoint(adapter_dir):
    if not adapter_dir.exists():
        return None, 0
    cps = sorted(adapter_dir.glob("*_adapters.safetensors"))
    if not cps:
        single = adapter_dir / "adapters.safetensors"
        return (single, 0) if single.exists() else (None, 0)
    try:
        return cps[-1], int(cps[-1].name.split("_")[0])
    except:
        return cps[-1], 0

def parse_line(line):
    global best_val_loss, best_iter
    if "Iter " in line and "Val loss" in line:
        try:
            it = int(line.split(":")[0].replace("Iter ", "").strip())
            vl = float(line.split("Val loss")[1].split(",")[0].strip())
            metrics.append({"iter": it, "val_loss": vl, "ts": datetime.now().isoformat()})
            save_metrics()
            if vl < best_val_loss:
                best_val_loss, best_iter = vl, it
                log(f"*** NEW BEST: iter={it}, val_loss={vl:.4f} ***")
            elif it % 1000 == 0:
                log(f"Val: {vl:.4f} (best: {best_val_loss:.4f} @ {best_iter})")
        except: pass
    if "Iter " in line and "Train loss" in line:
        try:
            it = int(line.split(":")[0].replace("Iter ", "").strip())
            if it % 500 == 0:
                log(f"Iter {it}: {line.strip()}")
        except: pass
    if "Saved" in line:
        log(f"Checkpoint: {line.strip()}")
    if "OutOfMemory" in line or "Insufficient Memory" in line:
        return False
    return True

def run_training(resume_from=None, start_iter=0):
    cmd = [sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL, "--train", "--data", str(DATA_DIR),
        "--adapter-path", str(ADAPTER_DIR),
        "--iters", str(TOTAL_ITERS - start_iter),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--steps-per-eval", str(EVAL_EVERY),
        "--save-every", str(SAVE_EVERY),
        "--num-layers", str(LORA_LAYERS),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--mask-prompt", "--seed", str(SEED), "--grad-checkpoint"]
    if resume_from:
        cmd.extend(["--resume-adapter-file", str(resume_from)])
    log(f"Batch={BATCH_SIZE}, iters={TOTAL_ITERS - start_iter}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1)

def main():
    global best_val_loss, best_iter, training_running
    log("=" * 60)
    log("BHARAT-TINY-LLM Bulletproof Training v2")
    log("=" * 60)
    last_cp, last_it = find_last_checkpoint(ADAPTER_DIR)
    if last_cp:
        log(f"Resuming from: {last_cp} (iter {last_it})")
    else:
        last_it = 0
        log("Starting from scratch")
    remaining = TOTAL_ITERS - last_it
    if remaining <= 0:
        log("Done!"); return
    log(f"Target: {TOTAL_ITERS} | Remaining: {remaining}")
    attempt = 0
    while attempt < 3 and training_running:
        attempt += 1
        log(f"\n--- Attempt {attempt} ---")
        proc = run_training(last_cp, last_it)
        oom = False
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line: continue
                if not parse_line(line):
                    oom = True; break
                if "Iter " in line and "Train loss" in line:
                    try: last_it = int(line.split(":")[0].replace("Iter ", "").strip())
                    except: pass
        except KeyboardInterrupt:
            log("Interrupted"); proc.terminate(); training_running = False; break
        rc = proc.wait()
        if oom:
            log("OOM! Retrying..."); last_cp, last_it = find_last_checkpoint(ADAPTER_DIR); continue
        if rc == 0:
            log("Training completed!"); break
        else:
            log(f"Exit code: {rc}"); last_cp, last_it = find_last_checkpoint(ADAPTER_DIR)
    log(f"\nBest val loss: {best_val_loss:.4f} @ iter {best_iter}")
    log(f"Adapter: {ADAPTER_DIR}")
    save_metrics()

def signal_handler(sig, frame):
    global training_running; log("Shutdown signal"); training_running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if LOG_FILE.exists(): LOG_FILE.unlink()
    main()
