#@title 1 · Setup {display-mode:"form"}
%pip install -q "transformers==4.51.3" peft accelerate datasets safetensors sentencepiece huggingface_hub 2>&1 | tail -1
# peft>=0.18 tries torchao LoRA dispatch; Kaggle ships torchao 0.10 which it
# rejects. We don't use torchao quant LoRA so remove it to avoid the clash.
%pip uninstall -y torchao 2>/dev/null; echo "torchao removed"

import json, math, os, re, signal, subprocess, sys, time, gzip, random
from collections import Counter
from pathlib import Path

import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub import (
    HfApi, create_repo, hf_hub_download, list_repo_files,
    snapshot_download, upload_file, upload_folder,
)

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)
api = HfApi(token=HF_TOKEN)

GPUS = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs:", GPUS)
assert torch.cuda.device_count() >= 1, "No GPU attached!"


def kill_session(reason=""):
    """Flush everything and hard-exit so Kaggle stops billing GPU time."""
    print(f"\n*** KILLING SESSION: {reason} ***\n", flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(2)
    finally:
        os.kill(os.getpid(), signal.SIGKILL)
