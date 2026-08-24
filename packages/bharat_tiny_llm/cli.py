"""`bharat` — offline Hindi/Hinglish chat on Apple Silicon.

Commands:
    bharat chat            Interactive REPL (streaming, conversation memory)
    bharat ask "question"  One-shot answer
    bharat info            Show model + package info

The model (~880 MB) auto-downloads from Hugging Face on first use and then
runs fully offline.
"""

import argparse
import os
import sys
from pathlib import Path

from .constants import (
    MODEL_V2_ID,
    MODEL_V2_LORA,
    GENERATION_CONFIG,
    GITHUB_URL,
)
from . import __version__

_BANNER = r"""
  ____  _                _
 | __ )| | __ _ _ __   __| |
 |  _ \| |/ _` | '_ \ / _` |
 | |_) | | (_| | | | | (_| |
 |____/|_|\__,_|_| |_|\__,_|   v{version}

 Offline Hindi/Hinglish assistant · {url}
 NOTE: no system prompt by default - v2 was trained without system turns.
"""


def _get_backend():
    try:
        from mlx_lm import load, stream_generate  # noqa: F401
        from mlx_lm.sample_utils import make_sampler  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_adapter(adapter: str | None) -> str | None:
    """Resolve an adapter spec to a local directory (HF repo subfolders are
    downloaded to the local HF cache on first use)."""
    import os
    from pathlib import Path

    spec = adapter or MODEL_V2_LORA
    if os.path.isdir(spec):
        return spec
    if "lora_adapter" in spec and not os.path.exists(spec):
        from huggingface_hub import snapshot_download

        repo_id = spec.split("/lora_adapter")[0]
        return snapshot_download(
            repo_id=repo_id, allow_patterns=["lora_adapter/*"]
        ) + "/lora_adapter"
    raise FileNotFoundError(f"adapter not found: {spec}")


def _load_model(adapter: str | None):
    from mlx_lm import load

    return load(os.environ.get("BHARAT_MODEL", MODEL_V2_ID),
                adapter_path=_resolve_adapter(adapter))


def _make_sampler():
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(
        temp=GENERATION_CONFIG["temperature"], top_p=GENERATION_CONFIG["top_p"]
    )


def _stream_reply(model, tokenizer, messages, max_tokens=None):
    from mlx_lm import stream_generate

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from .loaders import mlx_logits_processors

    text_out = []
    for resp in stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens or GENERATION_CONFIG["max_new_tokens"],
        sampler=_make_sampler(),
        logits_processors=mlx_logits_processors(),
    ):
        sys.stdout.write(resp.text)
        sys.stdout.flush()
        text_out.append(resp.text)
    print()
    return "".join(text_out)


def cmd_chat(args) -> int:
    if not _get_backend():
        print("MLX backend missing. Install with: pip install 'bharat-tiny-llm[mlx]'",
              file=sys.stderr)
        return 1

    print(_BANNER.format(version=__version__, url=GITHUB_URL))
    print("Loading model (first run downloads ~880 MB)...", flush=True)
    model, tokenizer = _load_model(args.adapter)
    print("Ready.\n")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    while True:
        try:
            user = input("आप › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nअलविदा!")
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = []
            print("(history cleared)\n")
            continue
        messages.append({"role": "user", "content": user})
        reply = _stream_reply(model, tokenizer, messages, max_tokens=args.max_tokens)
        messages.append({"role": "assistant", "content": reply})
    return 0


def cmd_ask(args) -> int:
    if not _get_backend():
        print("MLX backend missing. Install with: pip install 'bharat-tiny-llm[mlx]'",
              file=sys.stderr)
        return 1
    model, tokenizer = _load_model(args.adapter)
    messages = ([{"role": "system", "content": args.system}] if args.system else [])
    messages.append({"role": "user", "content": args.prompt})
    _stream_reply(model, tokenizer, messages, max_tokens=args.max_tokens)
    return 0


def cmd_info(_) -> int:
    print(f"bharat-tiny-llm {__version__}")
    print(f"model      : {MODEL_V2_ID}")
    print(f"adapter    : {MODEL_V2_LORA}")
    print(f"generation : {GENERATION_CONFIG}")
    print(f"backend    : {'mlx' if _get_backend() else 'NOT INSTALLED'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bharat",
        description="Offline Hindi/Hinglish assistant powered by Bharat-Tiny-LLM v2",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--adapter", default=None,
                       help="LoRA adapter path/repo (default: bundled v2 adapter)")
        p.add_argument("--max-tokens", type=int, default=None,
                       help=f"max new tokens (default {GENERATION_CONFIG['max_new_tokens']})")
        p.add_argument("--system", default=None, help="override system prompt")

    p_chat = sub.add_parser("chat", help="interactive chat REPL")
    add_common(p_chat)
    p_chat.set_defaults(fn=cmd_chat)

    p_ask = sub.add_parser("ask", help="one-shot question")
    p_ask.add_argument("prompt")
    add_common(p_ask)
    p_ask.set_defaults(fn=cmd_ask)

    p_info = sub.add_parser("info", help="show model/package info")
    p_info.set_defaults(fn=cmd_info)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
