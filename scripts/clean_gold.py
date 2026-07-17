"""
Gold-dataset cleaner for Bharat-Tiny-LLM Hinglish/Hindi fine-tuning.

Fixes identified in audit:
1. Cross-script contamination: drop samples where assistant turn has >=15 chars
   of a non-Devanagari / non-Latin script (Bengali, Telugu, Cyrillic, CJK, Arabic, Thai, Korean).
2. Echo samples: user content == assistant content (degenerate copies).
3. Over-represented prompts: cap each distinct prompt at N repetitions.
4. Normalize: Unicode NFC, strip control chars, strip mojibake.
5. Optionally keep bilingual (Hinglish + Devanagari) -- do NOT hard-filter Devanagari
   (that's Akshar's choice; our moat is bilingual).
6. Empty/garbage turns dropped.
"""
import json, re, unicodedata, sys
from collections import Counter, defaultdict

TRAIN = "/Users/eulogikdeveloper/Documents/Brahmi/data/processed/train.jsonl"
VALID = "/Users/eulogikdeveloper/Documents/Brahmi/data/processed/valid.jsonl"
OUT_TRAIN = "/Users/eulogikdeveloper/Documents/Brahmi/data/processed/train_gold.jsonl"
OUT_VALID = "/Users/eulogikdeveloper/Documents/Brahmi/data/processed/valid_gold.jsonl"

# Unicode ranges considered "allowed target scripts" for Hinglish/Hindi:
#   Latin (basic + Latin-1 supplement + Latin Extended)  -> A-Z a-z à-ÿ etc.
#   Devanagari                                    -> \u0900-\u097F
# Everything else (Bengali, Telugu, Tamil, Cyrillic, CJK, Arabic, Thai, Korean, etc.)
# is treated as contamination if it appears in >= CONTAM_THRESHOLD chars in the assistant turn.
ALLOWED = re.compile(
    r'[\u0000-\u024F\u0900-\u097F\u2000-\u206F\u20A0-\u20CF\u2190-\u21FF'
    r'\u2300-\u23FF\u2500-\u25FF\u2700-\u27BF\u3000-\u303F'  # CJK punctuation (acceptable)
    r'\uFE00-\uFE0F\u0300-\u036F'  # combining marks
    r']',
    re.UNICODE,
)
# Scripts we explicitly reject in assistant targets (contamination classes):
CONTAM_SCRIPTS = [
    (0x0980, 0x09FF, "Bengali"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x4E00, 0x9FFF, "CJK"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0E00, 0x0E7F, "Thai"),
    (0xAC00, 0xD7AF, "Korean"),
    (0x0C80, 0x0CDF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
]

CONTAM_THRESHOLD = 8   # chars of any single contaminant script in assistant turn
PROMPT_CAP = 10        # max repetitions of an identical prompt
MIN_ASSISTANT_WORDS = 2

def norm(text):
    if not isinstance(text, str):
        return ""
    # NFC normalization (kills mojibake/different byte forms of same char)
    text = unicodedata.normalize("NFC", text)
    # strip control chars (except tab/newline)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

def count_contam(text):
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, name in CONTAM_SCRIPTS:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return counts

def get_turns(msgs):
    user, asst, sysmsg = [], [], []
    for m in msgs:
        r = m.get("role", "")
        c = norm(m.get("content", ""))
        if r == "user":
            user.append(c)
        elif r == "assistant":
            asst.append(c)
        elif r == "system":
            sysmsg.append(c)
    return user, asst, sysmsg

def is_valid_sample(user, asst, sysmsg):
    # need at least one user turn and one non-empty assistant turn
    if not user or all(len(u) == 0 for u in user):
        return False, "no_user"
    if not asst or all(len(a) == 0 for a in asst):
        return False, "no_assistant"
    # echo: any user turn == any assistant turn
    for u in user:
        for a in asst:
            if u and u == a:
                return False, "echo"
    # assistant too short
    full_asst = " ".join(asst)
    if len(full_asst.split()) < MIN_ASSISTANT_WORDS:
        return False, "asst_short"
    # cross-script contamination in assistant target
    contam = count_contam(full_asst)
    for name, n in contam.items():
        if n >= CONTAM_THRESHOLD:
            return False, f"contam:{name}({n})"
    return True, "ok"

def process(inp, outp, prompt_cap, is_train):
    seen_prompts = Counter()
    stats = Counter()
    kept = 0
    total = 0
    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            msgs = d.get("messages", [])
            if not msgs:
                stats["no_messages"] += 1
                continue
            user, asst, sysmsg = get_turns(msgs)
            ok, reason = is_valid_sample(user, asst, sysmsg)
            if not ok:
                stats[reason] += 1
                continue
            # prompt cap (only for train split; valid kept fully clean)
            prompt_key = user[0]
            if is_train:
                seen_prompts[prompt_key] += 1
                if seen_prompts[prompt_key] > prompt_cap:
                    stats["prompt_capped"] += 1
                    continue
            # rebuild normalized sample
            new_msgs = []
            for m in msgs:
                new_msgs.append({"role": m.get("role", ""), "content": norm(m.get("content", ""))})
            g.write(json.dumps({"messages": new_msgs}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"\n[{inp}]")
    print(f"  total:   {total}")
    print(f"  kept:    {kept}")
    print(f"  dropped: {total - kept}")
    for k, v in stats.most_common():
        print(f"    - {k}: {v}")
    return kept

if __name__ == "__main__":
    process(TRAIN, OUT_TRAIN, PROMPT_CAP, is_train=True)
    process(VALID, OUT_VALID, PROMPT_CAP, is_train=False)
