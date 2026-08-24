"""
Bharat-Tiny-LLM Production Pipeline

Full end-to-end: corpus analysis → vocab injection → embedding expansion → 
MLX quantization → HF upload.

Usage:
  python scripts/pipeline.py [--budget 300] [--steps 0] [--upload]

Steps:
  1. Analyze corpus, select best Devanagari subwords
  2. Inject into Qwen2.5 tokenizer
  3. Expand model embeddings
  4. [Optional] Continued pre-training (GPU recommended)
  5. Convert to MLX Q4
  6. [Optional] Upload to Hugging Face

Requires: torch, transformers, mlx-lm
"""

import json, os, sys, math, time, re, argparse
from collections import Counter
from pathlib import Path

CACHE = "/Volumes/KIOXIA 1TB"
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
CORPUS = "data/processed/train_gold_v3.jsonl"


def extract_deva(text):
    runs = []
    run = ''
    for c in text:
        if 0x0900 <= ord(c) <= 0x097F:
            run += c
        else:
            if len(run) >= 2: runs.append(run)
            run = ''
    if len(run) >= 2: runs.append(run)
    return runs


def analyze(lines, max_n=6, sample=10000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    counts = {n: Counter() for n in range(2, max_n+1)}
    for line in lines[:sample]:
        data = json.loads(line)
        for msg in data['messages']:
            runs = extract_deva(msg['content'])
            for run in runs:
                for n in range(2, max_n+1):
                    for i in range(len(run)-n+1):
                        counts[n][run[i:i+n]] += 1
    candidates = []
    for n in range(2, max_n+1):
        for ng, freq in counts[n].most_common(2000):
            ids = tok.encode(ng)
            saved = len(ids) - 1
            if saved > 0:
                candidates.append((saved * math.sqrt(freq), saved, freq, ng, ids))
    candidates.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for score, saved, freq, ng, ids in candidates:
        if ng in seen: continue
        seen.add(ng)
        selected.append((ng, saved, freq, score, ids))
    return selected


def inject_and_expand(tokens, output_dir, init='mean'):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\nInjecting {len(tokens)} tokens...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    to_add = [t[0] for t in tokens]
    num_added = tok.add_tokens(to_add)
    added_ids = [tid for tid, t in sorted(tok.added_tokens_decoder.items())
                 if not str(t).startswith('<') and not getattr(t, 'special', False)]
    print(f"  → {num_added} tokens added, IDs {added_ids[0]}–{added_ids[-1]}")

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
    )
    old_vocab = model.config.vocab_size
    hidden = model.config.hidden_size
    new_size = old_vocab + len(added_ids)

    embed = model.model.embed_tokens.weight.data
    lmhead = model.lm_head.weight.data
    new_embed = torch.zeros(new_size, hidden, dtype=embed.dtype)
    new_lmhead = torch.zeros(new_size, hidden, dtype=lmhead.dtype)
    new_embed[:old_vocab] = embed
    new_lmhead[:old_vocab] = lmhead

    for i, (ng, saved, freq, _, ids) in enumerate(tokens):
        tid = added_ids[i]
        if init == 'mean':
            init_v = embed[ids].mean(dim=0)
        elif init == 'random':
            init_v = torch.randn(hidden, dtype=embed.dtype) * embed.std().item()
        else:
            init_v = embed[ids[0]]
        new_embed[tid] = init_v
        new_lmhead[tid] = init_v.clone()

    model.model.embed_tokens.weight = torch.nn.Parameter(new_embed)
    model.lm_head.weight = torch.nn.Parameter(new_lmhead)
    model.config.vocab_size = new_size

    print(f"  → Expanded vocab: {old_vocab} → {new_size} ({time.time()-t0:.0f}s)")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"  → Saved to {output_dir}")

    # Validate compression
    val_lines = open(CORPUS).readlines()[10000:15000]
    deva_texts = []
    for line in val_lines:
        data = json.loads(line)
        for msg in data['messages']:
            deva_texts.extend(extract_deva(msg['content']))

    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    chars = sum(len(t) for t in deva_texts)
    base_toks = sum(len(base_tok.encode(t)) for t in deva_texts[:500])
    new_toks = sum(len(tok.encode(t)) for t in deva_texts[:500])
    base_cpt = chars / base_toks if base_toks > 0 else 0
    new_cpt = chars / new_toks if new_toks > 0 else 0
    impr = ((new_cpt - base_cpt) / base_cpt * 100) if base_cpt > 0 else 0
    saved_pct = (1 - new_toks / base_toks) * 100 if base_toks > 0 else 0

    print(f"  → Compression: {base_cpt:.3f} → {new_cpt:.3f} chars/tok ({impr:.1f}% impr)")
    print(f"  → Token savings: {saved_pct:.1f}%")

    report = {
        "tokens_added": num_added,
        "token_ids": added_ids,
        "old_vocab_size": old_vocab,
        "new_vocab_size": new_size,
        "compression_chars_per_token_baseline": round(base_cpt, 3),
        "compression_chars_per_token_improved": round(new_cpt, 3),
        "compression_improvement_pct": round(impr, 1),
        "token_savings_pct": round(saved_pct, 1),
        "init_method": init,
        "selected_tokens": [{"token": t[0], "saved_tokens": t[1], "frequency": t[2]} for t in tokens],
    }
    with open(os.path.join(output_dir, "pipeline_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_dir


def to_mlx(pytorch_dir, mlx_dir, quantize=True):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mlx_lm import convert as mlx_convert

    print(f"\nConverting to MLX {'(Q4)' if quantize else ''}...")
    t0 = time.time()
    mlx_convert(
        hf_path=pytorch_dir,
        mlx_path=mlx_dir,
        quantize=quantize,
        q_group_size=64,
        q_bits=4,
        q_mode='affine',
    )
    print(f"  → Done in {time.time()-t0:.0f}s. Saved to {mlx_dir}")

    size = sum(f.stat().st_size for f in Path(mlx_dir).glob('*') if f.is_file())
    print(f"  → MLX model size: {size/1e9:.2f} GB")
    return mlx_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=300, help='Number of tokens to inject')
    parser.add_argument('--init', choices=['mean', 'random', 'first'], default='mean')
    parser.add_argument('--output', default=f'{CACHE}/brahmi-expanded')
    parser.add_argument('--mlx-output', default=f'{CACHE}/brahmi-mlx')
    parser.add_argument('--skip-inject', action='store_true')
    parser.add_argument('--skip-mlx', action='store_true')
    parser.add_argument('--upload', action='store_true', help='Upload to HF')
    parser.add_argument('--repo-name', default='eulogik/Bharat-Tiny-LLM-v2')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.mlx_output, exist_ok=True)

    if not args.skip_inject:
        print("=" * 50)
        print("STAGE 1: Corpus Analysis")
        print("=" * 50)
        lines = open(CORPUS).readlines()
        print(f"Corpus: {len(lines)} lines")
        selected = analyze(lines, sample=min(10000, len(lines)))
        print(f"Selected {len(selected)} candidate tokens")
        for i, (ng, saved, freq, score, ids) in enumerate(selected[:10]):
            print(f"  {i+1}. {ng} (saves {saved} tok, freq {freq})")

        print("\n" + "=" * 50)
        print("STAGE 2: Injection & Embedding Expansion")
        print("=" * 50)
        final_tokens = selected[:args.budget]
        inject_and_expand(final_tokens, args.output, args.init)

    if not args.skip_mlx:
        print("\n" + "=" * 50)
        print("STAGE 3: MLX Conversion (Q4)")
        print("=" * 50)
        to_mlx(args.output, args.mlx_output, quantize=True)

    if args.upload:
        print("\n" + "=" * 50)
        print("STAGE 4: Hugging Face Upload")
        print("=" * 50)
        from huggingface_hub import HfApi
        api = HfApi()
        for path, repo_id in [
            (args.output, args.repo_name + "-expanded"),
            (args.mlx_output, args.repo_name + "-MLX"),
        ]:
            if os.path.exists(path):
                print(f"Uploading {path} → {repo_id}...")
                api.upload_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  → https://huggingface.co/{repo_id}")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Expanded PyTorch model: {args.output}")
    print(f"MLX Q4 model: {args.mlx_output}")
    print(f"Report: {args.output}/pipeline_report.json")


if __name__ == '__main__':
    main()
