"""Convenience loaders for Bharat-Tiny-LLM.

The package only depends on ``huggingface_hub`` at install time. The actual
inference backend (``mlx-lm`` on Apple Silicon, or ``transformers`` elsewhere)
is imported lazily so the package stays lightweight.
"""

from .constants import MODEL_ID, GENERATION_CONFIG

_DEFAULT_BACKEND_MSG = (
    "No inference backend found. Install one:\n"
    "  Apple Silicon : pip install bharat-tiny-llm[mlx]\n"
    "  Other (CPU/GPU): pip install bharat-tiny-llm[torch]"
)


def load(model_id: str = MODEL_ID, backend: str | None = None):
    """Load the model + tokenizer.

    Parameters
    ----------
    model_id:
        Hugging Face repo id. Defaults to the Q4 edge model
        (``eulogik/Bharat-Tiny-LLM``), an Apple MLX 4-bit build.
    backend:
        ``"mlx"`` or ``"torch"``. Auto-detected from the platform when ``None``.

    Returns
    -------
    (model, tokenizer)
    """
    if backend is None:
        import platform

        backend = "mlx" if platform.system() == "Darwin" else "torch"

    if backend == "mlx":
        try:
            from mlx_lm.utils import load as _load
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_DEFAULT_BACKEND_MSG) from exc
        return _load(model_id)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_DEFAULT_BACKEND_MSG) from exc
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def chat(
    messages,
    model_id: str = MODEL_ID,
    backend: str | None = None,
    max_new_tokens: int | None = None,
):
    """Generate a reply for a list of ``{"role", "content"}`` messages.

    Uses the canonical :data:`GENERATION_CONFIG` so output is clean out of the
    box — no garbled scripts, no degenerate loops.
    """
    model, tokenizer = load(model_id, backend)

    if backend == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        import mlx.core as mx

        def _ngram_blocker(ngram_size=3):
            def _proc(tokens, logits):
                if len(tokens) < ngram_size:
                    return logits
                ngrams = {}
                for i in range(len(tokens) - ngram_size + 1):
                    ngrams.setdefault(tuple(tokens[i : i + ngram_size - 1]), []).append(
                        tokens[i + ngram_size - 1]
                    )
                prev = tuple(tokens[-(ngram_size - 1):])
                if prev in ngrams:
                    idx = mx.array(ngrams[prev])
                    logits = logits.at[idx].add(-1e9)
                return logits

            return _proc

        sampler = make_sampler(
            temp=GENERATION_CONFIG["temperature"], top_p=GENERATION_CONFIG["top_p"]
        )
        lps = [
            make_repetition_penalty(GENERATION_CONFIG["repetition_penalty"]),
            _ngram_blocker(GENERATION_CONFIG["no_repeat_ngram_size"]),
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        n = max_new_tokens or GENERATION_CONFIG["max_new_tokens"]
        return generate(
            model, tokenizer, prompt=prompt, max_tokens=n, sampler=sampler,
            logits_processors=lps,
        )

    import torch

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or GENERATION_CONFIG["max_new_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            no_repeat_ngram_size=GENERATION_CONFIG["no_repeat_ngram_size"],
            do_sample=GENERATION_CONFIG["do_sample"],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
