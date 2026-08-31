#@title 1 · Install deps {display-mode:"form"}
# Kaggle's preinstalled transformers 5.x + peft 0.17 already support qwen3.
# Only fix the peft ↔ torchao version clash (0.10 < required 0.16).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               capture_output=True)
print("torchao removed; deps ready")
