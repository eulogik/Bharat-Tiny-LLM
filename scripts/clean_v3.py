"""
Clean training data: remove rows with foreign-script contamination.

Domain = Hinglish (Latin) + Devanagari Hindi. Any other script
(Bengali, Tamil, Telugu, Cyrillic, CJK, Arabic, Thai, Korean, etc.)
is contamination and the row is dropped.

Input:  data/processed/train_gold_v2.jsonl
Output: data/processed/train_gold_v3.jsonl
"""
import json

SRC = "data/processed/train_gold_v2.jsonl"
DST = "data/processed/train_gold_v3.jsonl"

# Foreign scripts (outside our Hinglish/Devanagari domain)
FOREIGN = [
    (0x0980,0x09FF),(0x0C00,0x0C7F),(0x0B80,0x0BFF),(0x0400,0x04FF),(0x4E00,0x9FFF),
    (0x0600,0x06FF),(0x0E00,0x0E7F),(0xAC00,0xD7AF),(0x0C80,0x0CDF),(0x0D00,0x0D7F),
    (0x0A00,0x0A7F),(0x0B00,0x0B7F),(0x0F00,0x0FFF),(0x1780,0x17FF),(0x1000,0x109F),
    (0x0A80,0x0AFF),(0x1F00,0x1FFF),
]

def is_foreign(c):
    cp = ord(c)
    # allow Devanagari, Latin, digits, common punctuation/whitespace
    if 0x0900 <= cp <= 0x097F: return False   # Devanagari
    if 0x0041 <= cp <= 0x005A: return False   # Latin UC
    if 0x0061 <= cp <= 0x007A: return False   # Latin LC
    if 0x0030 <= cp <= 0x0039: return False   # digits
    if cp < 0x0100: return False               # ASCII punctuation/space
    for lo, hi in FOREIGN:
        if lo <= cp <= hi:
            return True
    return False

total = kept = dropped = 0
with open(SRC, encoding="utf-8") as fin, open(DST, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        d = json.loads(line)
        text = " ".join(m.get("content", "") for m in d.get("messages", []))
        if any(is_foreign(ch) for ch in text):
            dropped += 1
            continue
        fout.write(line)
        kept += 1

print(f"Total:   {total}")
print(f"Kept:    {kept}")
print(f"Dropped: {dropped} ({dropped*100//total}%)")
print(f"Wrote:   {DST}")
