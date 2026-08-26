#!/usr/bin/env python3
"""Assemble notebooks/v3_cells/*.py into notebooks/brahmi_v3_kaggle.ipynb.
Every code cell is syntax-validated (%%writefile cell checked after stripping magic)."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "notebooks" / "v3_cells"
OUT = ROOT / "notebooks" / "brahmi_v3_kaggle.ipynb"

MD_BLOCKS = {
    "c1_config.py": """\
# 🇮🇳 Brahmi v3 — Qwen3-1.7B + Token Injection + Capability SFT (Kaggle 2×T4)

**Self-healing, fully resumable.** All progress persists to the private HF repo
`eulogik/brahmi-v3-artifacts`. If a session dies or OOMs: **Runtime → Run all**
and it continues from where it stopped.

| Phase | What | Resume guard |
|---|---|---|
| 1 | Mine Devanagari n-grams vs **Qwen3** vocab → top-300 tokens (~30 min CPU) | `phase1` |
| 2 | Inject + mean-init embeddings → ~1 MB `embed_delta.safetensors` | `phase2` |
| 3 | Warmup new embedding rows only (3K steps, ~40 min) | `warmup_step` |
| 4 | Capability SFT — LoRA r32 DDP on 2×T4, OOM-shrink retries | checkpoints + heartbeat |
| 5 | Merge + upload `eulogik/Bharat-Tiny-LLM-v3` (fp16) + M4 handoff commands | `phase5` |
| DPO | Optional scaffold (`RUN_DPO`) | flag |

### Quota watchdog
- OOM → halves per-device batch, doubles accumulation, relaunches
- Attempt with **zero step progress** counts toward stall; ≥2 stalled attempts →
  state saved, session **killed** to stop burning GPU quota
- Hard deadline `MAX_SESSION_HOURS=10.5` (Kaggle cap is 12) → graceful stop + kill

### One-time setup
Attach a Kaggle secret named **HF_TOKEN** (write access), or paste it into the
CONFIG cell. Everything else is automatic.
""",
    "c8_runner.py": None,  # md injected between 6a and 6b handled below
}


def validate(name: str, src: str):
    if src.startswith("%%writefile"):
        body = "\n".join(src.splitlines()[1:])
        if body.startswith("\n"):
            body = body[1:]
    else:
        # strip IPython magics/param comments for AST check
        body = "\n".join(
            l for l in src.splitlines()
            if not l.strip().startswith(("%", "!", "@title"))
        )
    try:
        ast.parse(body)
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {name}: {e}")
        if e.text:
            print("   ", e.text.rstrip())
        sys.exit(1)


def main():
    order = ["c1_config.py", "c2_setup.py", "c3_state.py", "c4_mine.py",
             "c5_inject.py", "c6_warmup.py", "c7_train_sft.py",
             "c8_runner.py", "c9_export.py", "c10_dpo.py", "c11_summary.py"]
    present = {p.name for p in CELLS.glob("*.py")}
    missing = [f for f in order if f not in present]
    extra = [f for f in present if f not in order]
    assert not missing, f"missing cells: {missing}"
    assert not extra, f"unexpected cells not in order list: {extra}"
    cells = []

    for name in order:
        raw = (CELLS / name).read_text()
        validate(name, raw)

        if name == "c1_config.py":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": MD_BLOCKS["c1_config.py"].splitlines(keepends=True),
            })

        first = raw.splitlines()[0]
        meta = {}
        if first.startswith("#@title"):
            title = first.replace("#@title", "").strip()
            hide = ' {display-mode:"form"}' in title
            title = title.replace(' {display-mode:"form"}', "")
            meta = {"cellView": "form" if hide else "form"}
            src = raw.splitlines(keepends=True)
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {"collapsed": "true", "_kg-hide-input": False},
                          "outputs": [], "source": src})
        else:
            src_lines = raw.splitlines(keepends=True)
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": src_lines})

    # split runner file into two cells at the 6b marker
    final_cells = []
    for c in cells:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        marker = "#@title 6b · PHASE 4 — Self-healing SFT runner on 2×T4"
        if marker in src:
            a, b = src.split(marker, 1)
            final_cells.append({"cell_type": "code", "execution_count": None,
                                "metadata": {"collapsed": "true"}, "outputs": [],
                                "source": a.rstrip() + "\n"})
            final_cells.append({"cell_type": "markdown", "metadata": {},
                                "source": ["## Phase 4b — Self-healing runner\n",
                                           "Launches `train_sft.py` under torchrun "
                                           "(DDP 2×T4). OOM → shrink batch & retry. "
                                           "No progress across attempts → kills "
                                           "session to save quota.\n"]})
            final_cells.append({"cell_type": "code", "execution_count": None,
                                "metadata": {"collapsed": "true"}, "outputs": [],
                                "source": ("#@title 6b · Self-healing SFT runner "
                                           "{display-mode:\"form\"}\n" + b)})
        else:
            final_cells.append(c)

    nb = {
        "cells": final_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    n_code = sum(1 for c in final_cells if c["cell_type"] == "code")
    print(f"wrote {OUT} ({n_code} code cells, all syntax-validated)")


if __name__ == "__main__":
    main()
