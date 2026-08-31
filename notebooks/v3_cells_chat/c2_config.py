#@title 1 · Config {display-mode:"form"}
import os, json, time, random, gzip
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────
WORK      = Path("/kaggle/working/brahmi")
DATA_DIR  = WORK / "data"
OUT_DIR   = WORK / "chat_sft_out"

# ── HF Token ──────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"kaggle_secrets failed: {e}")
if not HF_TOKEN:
    HF_TOKEN = "***REMOVED_HF_TOKEN***"
    print("WARNING: using hardcoded HF_TOKEN")

# Set as env var so huggingface_hub picks it up for ALL internal requests
# (transformers 5.x calls model_info() without forwarding token=)
os.environ["HF_TOKEN"] = HF_TOKEN
print(f"HF_TOKEN: ...{HF_TOKEN[-4:]}")

# ── Model ────────────────────────────────────────────────────────
BASE_MODEL = "eulogik/Bharat-Tiny-LLM-v3"    # already-merged v3
CHAT_DATA  = "eulogik/bharat-v3-chat-sft"     # our 20K chat dataset
OUT_REPO   = "eulogik/Bharat-Tiny-LLM-v3"    # overwrite v3 with v3.1

# ── Hyperparameters (T4 single-GPU) ────────────────────────────
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
BS            = 4          # per-device batch size
GRAD_ACCUM    = 4          # effective batch = 16 (fewer accum steps = faster)
EPOCHS        = 2
LR            = 2e-5       # conservative for chat SFT
MAX_SEQ       = 512
SAVE_STEPS    = 250
LOG_STEPS     = 25
MAX_SESSION_HOURS = 5.5

# ── State ────────────────────────────────────────────────────────
STATE_PATH = WORK / "chat_sft_state.json"
WORK.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_state():
    try: return json.loads(STATE_PATH.read_text())
    except: return {}

def save_state(s):
    STATE_PATH.write_text(json.dumps(s, indent=2))

STATE = load_state()
print(f"config ready | base={BASE_MODEL} | data={CHAT_DATA}")
print(f"lora r={LORA_R} alpha={LORA_ALPHA} | bs={BS} accum={GRAD_ACCUM} eff_bs={BS*GRAD_ACCUM}")
print(f"epochs={EPOCHS} lr={LR} seq={MAX_SEQ}")
