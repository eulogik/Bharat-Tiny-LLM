# Hinglish token injection — mining report

- Tokenizer: `Qwen/Qwen3.5-2B`
- Corpus: `/Volumes/KIOXIA 1TB/brahmi_backup/train_gold_v3.jsonl`
- Held-out: `data/processed/valid_gold.jsonl`
- Messages seen: **403,649**
- Vocab: **248,077 → 249,077** (+1000)
- Budget: 1000 (Hinglish 800, Devanagari 200)

## Held-out token savings

| Slice | Baseline tok/1000 chars | Injected tok/1000 chars | Savings |
|---|---:|---:|---:|
| Hinglish messages | 299.59 | 249.99 | 16.56% |
| All messages | 359.98 | 333.32 | 7.41% |
| English-only messages | 244.04 | 243.52 | 0.21% |
| Devanagari runs | 421.12 | 411.67 | 2.24% |

## Top Hinglish candidates

| # | Token | Current tokens | Freq | Saved/use | Score |
|---:|---|---:|---:|---:|---:|
| 1 | ` hain` | 2 | 188,032 | 1 | 434 |
| 2 | ` ekdum` | 3 | 33,911 | 2 | 368 |
| 3 | ` chahiye` | 3 | 28,228 | 2 | 336 |
| 4 | ` toh` | 2 | 104,966 | 1 | 324 |
| 5 | ` yeh` | 2 | 91,123 | 1 | 302 |
| 6 | ` bhi` | 2 | 63,218 | 1 | 251 |
| 7 | ` kya` | 2 | 61,286 | 1 | 248 |
| 8 | ` liye` | 2 | 51,998 | 1 | 228 |
| 9 | ` bahut` | 2 | 47,897 | 1 | 219 |
| 10 | ` samajh` | 3 | 11,148 | 2 | 211 |
| 11 | ` haanji` | 3 | 10,809 | 2 | 208 |
| 12 | ` kaise` | 2 | 40,618 | 1 | 202 |
| 13 | ` aapko` | 3 | 10,146 | 2 | 201 |
| 14 | ` karte` | 2 | 36,244 | 1 | 190 |
| 15 | ` rahega` | 3 | 8,490 | 2 | 184 |
| 16 | ` zaroori` | 3 | 7,910 | 2 | 178 |
| 17 | ` nahi` | 2 | 31,581 | 1 | 178 |
| 18 | ` hota` | 2 | 29,505 | 1 | 172 |
| 19 | ` haan` | 2 | 28,740 | 1 | 170 |
| 20 | ` dekho` | 2 | 26,426 | 1 | 163 |
| 21 | ` karna` | 2 | 26,426 | 1 | 163 |
| 22 | ` karne` | 2 | 26,365 | 1 | 162 |
| 23 | ` bilkul` | 2 | 25,887 | 1 | 161 |
| 24 | ` hinglish` | 2 | 25,798 | 1 | 161 |
| 25 | ` karo` | 2 | 24,856 | 1 | 158 |
| 26 | ` cheezein` | 3 | 6,133 | 2 | 157 |
| 27 | ` nahin` | 2 | 24,162 | 1 | 155 |
| 28 | ` sakte` | 2 | 24,042 | 1 | 155 |
| 29 | ` karein` | 3 | 5,651 | 2 | 150 |
| 30 | ` yahaan` | 3 | 5,519 | 2 | 149 |
| 31 | ` jaise` | 2 | 22,044 | 1 | 148 |
| 32 | ` zyada` | 2 | 22,016 | 1 | 148 |
| 33 | ` rahe` | 2 | 21,851 | 1 | 148 |
| 34 | ` hoon` | 2 | 21,384 | 1 | 146 |
| 35 | ` alag-alag` | 4 | 2,242 | 3 | 142 |
| 36 | ` sabse` | 2 | 19,260 | 1 | 139 |
| 37 | ` samajhna` | 4 | 2,045 | 3 | 136 |
| 38 | ` baat` | 2 | 18,311 | 1 | 135 |
| 39 | ` pehle` | 2 | 17,958 | 1 | 134 |
| 40 | ` aapne` | 3 | 4,395 | 2 | 133 |

_Devanagari candidates selected: 200_
