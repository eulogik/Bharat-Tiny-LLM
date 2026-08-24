from collections import Counter
from transformers import AutoTokenizer
import json
import re
import sys

def analyze_dev_n_grams(lines, n=5):
    counts = Counter()
    for line in lines:
        for msg in json.loads(line)['messages']:
            text = msg['content']
            for c in text:
                if not (0x0900 <= ord(c) <= 0x097F):
                    text = text.replace(c, ' ')
            parts = re.split(r'\s+', text)
            for part in parts:
                if len(part) < 2:
                    continue
                for i in range(len(part) - n + 1):
                    counts[part[i:i+n]] += 1
    return counts

def measure_compression(tok, texts):
    total_chars = 0
    total_tokens = 0
    for t in texts:
        ids = tok.encode(t)
        total_chars += len(t)
        total_tokens += len(ids)
    return total_chars / total_tokens if total_tokens > 0 else 0

def main():
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    baseline_tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')

    print(f"Baseline vocab: {tok.vocab_size}", file=sys.stderr)

    with open('data/processed/train_gold_v3.jsonl') as f:
        all_lines = f.readlines()

    sample = all_lines[:5000]

    print(f"Analyzing {len(sample)} lines...", file=sys.stderr)

    bigram_counts = analyze_dev_n_grams(sample, n=2)
    trigram_counts = analyze_dev_n_grams(sample, n=3)
    fourgram_counts = analyze_dev_n_grams(sample, n=4)
    fivegram_counts = analyze_dev_n_grams(sample, n=5)

    print(f"Bigrams: {len(bigram_counts)} unique", file=sys.stderr)
    print(f"Trigrams: {len(trigram_counts)} unique", file=sys.stderr)
    print(f"4-grams: {len(fourgram_counts)} unique", file=sys.stderr)
    print(f"5-grams: {len(fivegram_counts)} unique", file=sys.stderr)

    baseline_texts = []
    for line in all_lines[:2000]:
        for msg in json.loads(line)['messages']:
            text = msg['content']
            deva_parts = re.findall(r'[\u0900-\u097F]{5,}', text)
            baseline_texts.extend(deva_parts)

    baseline_cpt = measure_compression(baseline_tok, baseline_texts)
    print(f"\nBASELINE chars/token on Devanagari runs: {baseline_cpt:.3f}", file=sys.stderr)
    # chars per token < 1 means WORSE than 1:1
    print(f"BASELINE tokens/char: {1/baseline_cpt if baseline_cpt > 0 else 0:.3f}", file=sys.stderr)

    candidates = []
    for ng, cnt in fourgram_counts.most_common(200):
        if cnt < 5:
            continue
        baseline_ids = baseline_tok.encode(ng)
        saved_tokens = len(baseline_ids) - 1
        if saved_tokens > 0:
            candidates.append((saved_tokens, cnt, ng, baseline_ids))

    candidates.sort(key=lambda x: (-x[0], -x[1]))

    print(f"\nTop 30 Devanagari subwords by token savings:", file=sys.stderr)
    print(f"{'Rank':<5} {'Token':<15} {'Frequency':<10} {'Current tokens':<15} {'Saved tokens':<12} {'CPT improv':<10}", file=sys.stderr)
    for rank, (saved, cnt, ng, ids) in enumerate(candidates[:30], 1):
        current_len = len(ids)
        cpt_improvement = len(ng) / (current_len - saved)
        print(f"{rank:<5} {ng:<15} {cnt:<10} {current_len:<15} {saved:<12} {cpt_improvement:<10.2f}", file=sys.stderr)

    to_add = []
    for saved, cnt, ng, ids in candidates[:100]:
        if saved >= 1:
            to_add.append(ng)

    print(f"\nAdding {len(to_add)} new tokens...", file=sys.stderr)

    tok.add_tokens(to_add)

    improved_cpt = measure_compression(tok, baseline_texts)
    print(f"IMPROVED chars/token on Devanagari runs: {improved_cpt:.3f}", file=sys.stderr)
    print(f"IMPROVED tokens/char: {1/improved_cpt if improved_cpt > 0 else 0:.3f}", file=sys.stderr)

    improvement_pct = ((improved_cpt - baseline_cpt) / baseline_cpt) * 100 if baseline_cpt > 0 else 0
    print(f"Compression improvement: {improvement_pct:.1f}%", file=sys.stderr)

    tok.save_pretrained('/tmp/qwen_dev_tokenizer')
    print(f"\nTokenizer saved to /tmp/qwen_dev_tokenizer", file=sys.stderr)
    print(f"New tokenizer total vocab (base + added): {tok.vocab_size + len(tok.added_tokens_decoder)}", file=sys.stderr)

    print("\n--- JSON results ---")
    print(json.dumps({
        "baseline_cpt": round(baseline_cpt, 3),
        "improved_cpt": round(improved_cpt, 3),
        "tokens_per_char_baseline": round(1/baseline_cpt, 3) if baseline_cpt > 0 else 0,
        "tokens_per_char_improved": round(1/improved_cpt, 3) if improved_cpt > 0 else 0,
        "improvement_pct": round(improvement_pct, 1),
        "tokens_added": len(to_add),
        "top_candidates": [
            {"token": ng, "saved_tokens": saved, "frequency": cnt}
            for saved, cnt, ng, ids in candidates[:30]
        ]
    }))

if __name__ == '__main__':
    main()
