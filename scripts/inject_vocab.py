"""
Brahmic vocabulary injection for Qwen2.5-1.5B.

Stage 1: Analyze Devanagari subword frequency, select optimal tokens
Stage 2: Inject into tokenizer
Stage 3: Expand model embeddings
Stage 4: Validate compression
"""

import json
import re
import sys
import os
import math
from collections import Counter
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_deva_runs(text: str) -> List[str]:
    runs = []
    run = ''
    for c in text:
        if 0x0900 <= ord(c) <= 0x097F:
            run += c
        else:
            if len(run) >= 2:
                runs.append(run)
            run = ''
    if len(run) >= 2:
        runs.append(run)
    return runs


def count_dev_subwords(lines, max_n=6):
    """Count Devanagari character n-grams from corpus."""
    counts_by_n = {}
    for n in range(2, max_n + 1):
        counts_by_n[n] = Counter()

    for line in lines:
        data = json.loads(line)
        for msg in data['messages']:
            text = msg['content']
            runs = extract_deva_runs(text)
            for run in runs:
                for n in range(2, max_n + 1):
                    for i in range(len(run) - n + 1):
                        counts_by_n[n][run[i:i+n]] += 1
    return counts_by_n


def select_candidates(tok, counts_by_n, budget=200):
    """
    Select the best Devanagari subwords to inject.
    Prioritizes by token savings * frequency.
    """
    candidates = []

    for n in range(2, 7):
        for ng, freq in counts_by_n[n].most_common(2000):
            ids = tok.encode(ng)
            saved = len(ids) - 1
            if saved > 0:
                # Score: saved tokens * frequency (simulates total token savings)
                # We use sqrt(freq) to avoid over-indexing on very common words
                score = saved * math.sqrt(freq)
                candidates.append((score, saved, freq, ng, ids))

    candidates.sort(key=lambda x: -x[0])

    selected = []
    seen = set()
    for score, saved, freq, ng, ids in candidates:
        if ng in seen:
            continue
        seen.add(ng)
        selected.append((ng, saved, freq, score, ids))
        if len(selected) >= budget:
            break

    return selected


def expand_embeddings(model, tokenizer, added_token_ids: List[int], init_method='mean'):
    """Expand model embeddings for new tokens."""
    old_vocab_size = model.config.vocab_size
    new_vocab_size = old_vocab_size + len(added_token_ids)
    hidden_size = model.config.hidden_size

    old_embed = model.model.embed_tokens.weight.data
    old_lm_head = model.lm_head.weight.data

    new_embed = torch.zeros(new_vocab_size, hidden_size, dtype=old_embed.dtype)
    new_lm_head = torch.zeros(new_vocab_size, hidden_size, dtype=old_lm_head.dtype)

    new_embed[:old_vocab_size] = old_embed
    new_lm_head[:old_vocab_size] = old_lm_head

    if init_method == 'mean':
        for new_id, (ng, _, _, _, ids) in zip(added_token_ids, added_tokens_info):
            constituent_embeds = old_embed[ids]
            init_embed = constituent_embeds.mean(dim=0)
            new_embed[new_id] = init_embed
            new_lm_head[new_id] = init_embed.clone()
    elif init_method == 'random':
        embed_std = old_embed.std().item()
        for new_id in added_token_ids:
            new_embed[new_id] = torch.randn(hidden_size, dtype=old_embed.dtype) * embed_std
            new_lm_head[new_id] = torch.randn(hidden_size, dtype=old_lm_head.dtype) * embed_std

    model.model.embed_tokens.weight = torch.nn.Parameter(new_embed)
    model.lm_head.weight = torch.nn.Parameter(new_lm_head)
    model.config.vocab_size = new_vocab_size

    return model, new_vocab_size


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    init_method = sys.argv[2] if len(sys.argv) > 2 else 'mean'
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'Qwen/Qwen2.5-1.5B'
    corpus_path = sys.argv[4] if len(sys.argv) > 4 else 'data/processed/train_gold_v3.jsonl'
    output_dir = sys.argv[5] if len(sys.argv) > 5 else 'models/dev_expanded'

    print(f"Budget: {budget} tokens", file=sys.stderr)
    print(f"Init method: {init_method}", file=sys.stderr)
    print(f"Model: {model_name}", file=sys.stderr)
    print(f"Corpus: {corpus_path}", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Stage 1: Analyzing corpus ===", file=sys.stderr)
    lines = open(corpus_path).readlines()
    sample = lines[:min(10000, len(lines))]
    print(f"Loaded {len(lines)} lines, sampling {len(sample)}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(model_name)
    baseline_tok = AutoTokenizer.from_pretrained(model_name)

    counts_by_n = count_dev_subwords(sample, max_n=6)
    for n in range(2, 7):
        print(f"  {n}-grams: {len(counts_by_n[n])} unique", file=sys.stderr)

    print("\n=== Stage 2: Selecting candidates ===", file=sys.stderr)
    selected = select_candidates(baseline_tok, counts_by_n, budget=budget)

    print(f"Selected {len(selected)} tokens", file=sys.stderr)
    print(f"{'Rank':<5} {'Token':<15} {'Saved':<7} {'Freq':<8} {'Score':<8}", file=sys.stderr)
    for i, (ng, saved, freq, score, ids) in enumerate(selected[:30], 1):
        print(f"{i:<5} {ng:<15} {saved:<7} {freq:<8} {score:<8.0f}", file=sys.stderr)

    to_add = [ng for ng, _, _, _, _ in selected]

    print(f"\n=== Stage 3: Injecting tokens ===", file=sys.stderr)
    num_added = tok.add_tokens(to_add)
    print(f"Added {num_added} tokens to tokenizer", file=sys.stderr)

    added_token_ids = list(range(tok.vocab_size, tok.vocab_size + num_added))
    print(f"New token IDs: {added_token_ids[0]} to {added_token_ids[-1]}", file=sys.stderr)

    print(f"\n=== Stage 4: Expanding model embeddings ===", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    print(f"Original vocab_size: {model.config.vocab_size}", file=sys.stderr)

    global added_tokens_info
    added_tokens_info = [(ng, saved, freq, score, ids) for ng, saved, freq, score, ids in selected]

    model, new_vocab_size = expand_embeddings(model, tok, added_token_ids, init_method=init_method)
    print(f"New vocab_size: {new_vocab_size}", file=sys.stderr)

    print(f"\n=== Stage 5: Saving ===", file=sys.stderr)
    tok.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved to {output_dir}", file=sys.stderr)

    print(f"\n=== Stage 6: Compression Validation ===", file=sys.stderr)
    val_lines = lines[10000:15000]
    deva_texts = []
    for line in val_lines:
        data = json.loads(line)
        for msg in data['messages']:
            text = msg['content']
            runs = extract_deva_runs(text)
            deva_texts.extend(runs)

    if deva_texts:
        total_chars = sum(len(t) for t in deva_texts)
        total_tokens_baseline = sum(len(baseline_tok.encode(t)) for t in deva_texts)
        total_tokens_new = sum(len(tok.encode(t)) for t in deva_texts)

        baseline_cpt = total_chars / total_tokens_baseline if total_tokens_baseline > 0 else 0
        improved_cpt = total_chars / total_tokens_new if total_tokens_new > 0 else 0

        print(f"Validation texts: {len(deva_texts)}", file=sys.stderr)
        print(f"Total chars: {total_chars}", file=sys.stderr)
        print(f"Baseline tokens: {total_tokens_baseline}", file=sys.stderr)
        print(f"Improved tokens: {total_tokens_new}", file=sys.stderr)
        print(f"Baseline chars/token: {baseline_cpt:.3f}", file=sys.stderr)
        print(f"Improved chars/token: {improved_cpt:.3f}", file=sys.stderr)
        improvement = ((improved_cpt - baseline_cpt) / baseline_cpt) * 100
        print(f"Compression improvement: {improvement:.1f}%", file=sys.stderr)
        print(f"Tokens saved: {total_tokens_baseline - total_tokens_new} "
              f"({(1 - total_tokens_new/total_tokens_baseline)*100:.1f}%)", file=sys.stderr)
    else:
        print("No Devanagari texts found in validation set", file=sys.stderr)

    with open(os.path.join(output_dir, 'injection_report.json'), 'w') as f:
        report = {
            "budget": budget,
            "init_method": init_method,
            "model": model_name,
            "corpus": corpus_path,
            "tokens_added": num_added,
            "new_vocab_size": new_vocab_size,
            "baseline_chars_per_token": round(baseline_cpt, 3) if deva_texts else None,
            "improved_chars_per_token": round(improved_cpt, 3) if deva_texts else None,
            "compression_improvement_pct": round(improvement, 1) if deva_texts else None,
            "selected_tokens": [
                {"token": ng, "saved_tokens": saved, "frequency": freq, "decomposition": ids}
                for ng, saved, freq, _, ids in selected
            ],
        }
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Injection report at {output_dir}/injection_report.json", file=sys.stderr)


if __name__ == '__main__':
    main()
