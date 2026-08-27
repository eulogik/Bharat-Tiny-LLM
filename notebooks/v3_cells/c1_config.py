#@title 0 · CONFIG {display-mode:"form"}
HF_TOKEN = ""  # or attach Kaggle secret named HF_TOKEN
BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
ARTIFACTS_REPO = "eulogik/brahmi-v3-artifacts"
OUT_REPO = "eulogik/Bharat-Tiny-LLM-v3"

CORPUS_CANDIDATES = [
    ("eulogik/Bharat-Tiny-LLM-v2", "train_gold_v3.jsonl.gz", "model"),
    ("eulogik/Bharat-Tiny-LLM-v2", "train_gold_v3.jsonl.gz", "dataset"),
]

# phase 1-2
BUDGET = 300
MINE_LINES = 12000
VAL_LINES_OFFSET = 300000
VAL_LINES_COUNT = 2000

# phase 3 warmup
WARMUP_STEPS = 3000
WARMUP_BS = 4
WARMUP_SEQ = 256
WARMUP_LR = 1e-3

# phase 4 SFT
MAX_SFT_ROWS = 50000
SFT_EPOCHS = 2
SFT_LR = 1e-4
SFT_PER_DEVICE_BS = 4
SFT_GRAD_ACCUM = 2
SFT_SEQ = 1024
MIN_PER_DEVICE_BS = 1
MAX_ATTEMPTS = 4
STALL_LIMIT = 4
MAX_SESSION_HOURS = 10.5

RUN_DPO = False

import os
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
assert HF_TOKEN, "Set HF_TOKEN in CONFIG or attach Kaggle secret 'HF_TOKEN'"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("CONFIG ok | budget:", BUDGET, "| base:", BASE_MODEL)
