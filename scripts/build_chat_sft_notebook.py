#!/usr/bin/env python3
"""Assemble notebooks/v3_cells_chat/*.py into notebooks/brahmi_v3_chat_sft.ipynb."""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "notebooks" / "v3_cells_chat"
OUT = ROOT / "notebooks" / "brahmi_v3_chat_sft.ipynb"

ORDER = ["c1_install.py", "c2_config.py", "c3_load_model.py",
         "c4_tokenize.py", "c5_train.py", "c6_export.py"]

MD = """\
# 🇮🇳 Bharat-Tiny-LLM v3.1 — Conversational SFT (Kaggle T4)

Fine-tunes the **already-merged v3 model** (`eulogik/Bharat-Tiny-LLM-v3`)
on 20K Hindi/Hinglish conversations for natural chat ability.

| Cell | What | Resume guard |
|---|---|---|
| 1 | Install deps | — |
| 2 | Config | — |
| 3 | Load v3 + LoRA r=16 | `model_loaded` |
| 4 | Load & tokenize chat dataset | `tok_done` |
| 5 | Train (2 epochs, lr=2e-5) | `train_done` |
| 6 | Merge adapter + upload v3.1 | `export_done` |

### Dataset
- `eulogik/bharat-v3-chat-sft` (20K rows)
  - 10K multi-turn Hindi (`samvaad-hi-v1`)
  - 5K Hinglish (`hinglish-conversations`)
  - 5K filtered single-turn from our corpus

### One-time setup
Attach a Kaggle secret named **HF_TOKEN** (write access).
"""

def validate(name, src):
    if src.startswith("%%writefile"):
        body = "\n".join(src.splitlines()[1:])
    else:
        body = "\n".join(
            l for l in src.splitlines()
            if not l.strip().startswith(("%", "!", "@title"))
        )
    try:
        ast.parse(body)
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {name}: {e}")
        sys.exit(1)

def main():
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": MD.splitlines(keepends=True)}]
    for name in ORDER:
        raw = (CELLS / name).read_text()
        validate(name, raw)
        src = raw.splitlines(keepends=True)
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [], "source": src})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                       "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 4}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT} ({len(cells)-1} code cells)")

if __name__ == "__main__":
    main()
