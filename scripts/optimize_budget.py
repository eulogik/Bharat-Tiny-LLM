"""
Find the optimal Devanagari vocabulary injection budget.
Tests different budget sizes and measures:
- Compression improvement
- Token savings
- Diminishing returns curve
"""

import json
import re
import math
import sys
from collections import Counter
from typing import List, Tuple
import numpy as np

from transformers import AutoTokenizer


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
    candidates = []
    for n in range(2, 7):
        for ng, freq in counts_by_n[n].most_common(2000):
            ids = tok.encode(ng)
            saved = len(ids) - 1
            if saved > 0:
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


def measure_cpt(tok, texts):
    if not texts:
        return 0
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tok.encode(t)) for t in texts)
    return total_chars / total_tokens if total_tokens > 0 else 0


def main():
    corpus_path = 'data/processed/train_gold_v3.jsonl'

    lines = open(corpus_path).readlines()
    sample = lines[:10000]
    val_lines = lines[10000:15000]

    print(f"Total corpus: {len(lines)} lines", file=sys.stderr)
    print(f"Analysis sample: {len(sample)} lines", file=sys.stderr)
    print(f"Validation set: {len(val_lines)} lines", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')

    print("\n=== Analyzing corpus n-grams ===", file=sys.stderr)
    counts_by_n = count_dev_subwords(sample, max_n=6)
    for n in range(2, 7):
        print(f"  {n}-grams: {len(counts_by_n[n])} unique", file=sys.stderr)

    print("\n=== Building validation set ===", file=sys.stderr)
    deva_texts = []
    for line in val_lines:
        data = json.loads(line)
        for msg in data['messages']:
            text = msg['content']
            runs = extract_deva_runs(text)
            deva_texts.extend(runs)
    print(f"Validation Devanagari runs: {len(deva_texts)}", file=sys.stderr)

    baseline_cpt = measure_cpt(tok, deva_texts)
    baseline_tokens = sum(len(tok.encode(t)) for t in deva_texts)
    print(f"Baseline: {baseline_cpt:.3f} chars/token, {baseline_tokens} tokens", file=sys.stderr)

    budgets = [50, 100, 200, 300, 500, 750, 1000]

    print(f"\n{'Budget':<8} {'Tokens':<10} {'CPT':<8} {'Improvement':<12} {'Saved':<10} {'DimRet':<10}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    prev_improvement = 0
    results = []

    for budget in budgets:
        selected = select_candidates(tok, counts_by_n, budget=budget)
        to_add = [ng for ng, _, _, _, _ in selected]

        test_tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
        test_tok.add_tokens(to_add)

        improved_cpt = measure_cpt(test_tok, deva_texts)
        improved_tokens = sum(len(test_tok.encode(t)) for t in deva_texts)
        improvement = ((improved_cpt - baseline_cpt) / baseline_cpt) * 100
        dim_ret = improvement - prev_improvement
        saved_pct = (1 - improved_tokens / baseline_tokens) * 100

        results.append({
            'budget': budget,
            'baseline_cpt': round(baseline_cpt, 3),
            'improved_cpt': round(improved_cpt, 3),
            'improvement_pct': round(improvement, 2),
            'tokens_saved_pct': round(saved_pct, 2),
            'baseline_tokens': baseline_tokens,
            'improved_tokens': improved_tokens,
            'incremental_improvement': round(dim_ret, 2),
        })

        print(f"{budget:<8} {improved_tokens:<10} {improved_cpt:<8.3f} +{improvement:<9.2f}% {saved_pct:<8.2f}% {dim_ret:<+9.2f}%", file=sys.stderr)

        prev_improvement = improvement

        # Budget-per-token efficiency
        tokens_added = len(to_add)
        efficiency = improvement / tokens_added
        print(f"  → {efficiency:.3f}% improvement per token added", file=sys.stderr)

    print("\n\n=== Summary ===", file=sys.stderr)
    print(f"{'Budget':<8} {'CPT':<8} {'Improvement':<12} {'Tokens Saved':<14} {'Incremental':<12} {'Efficiency':<10}", file=sys.stderr)
    print("-" * 65, file=sys.stderr)
    for r in results:
        b = r['budget']
        eff = r['improvement_pct'] / b
        print(f"{b:<8} {r['improved_cpt']:<8.3f} +{r['improvement_pct']:<9.2f}% {r['tokens_saved_pct']:<11.2f}% +{r['incremental_improvement']:<8.2f}% {eff:<8.3f}%/tok", file=sys.stderr)

    # Find the knee point (best value per token)
    efficiencies = [(r['budget'], r['improvement_pct'] / r['budget']) for r in results]
    best_eff = max(efficiencies, key=lambda x: x[1])
    print(f"\nBest efficiency: {best_eff[0]} tokens @ {best_eff[1]:.3f}% improvement/token", file=sys.stderr)

    # Also find where incremental gain drops below threshold
    for i, r in enumerate(results):
        if i > 0 and r['incremental_improvement'] < 3.0:
            print(f"Knee point: {r['budget']} tokens (incremental gain {r['incremental_improvement']:.2f}%)", file=sys.stderr)
            break

    with open('models/budget_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to models/budget_analysis.json", file=sys.stderr)


if __name__ == '__main__':
    main()
