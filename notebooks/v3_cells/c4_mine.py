#@title 3 · PHASE 1 — Mine top-300 Devanagari subwords vs Qwen3 vocab (~30 min CPU) {display-mode:"form"}
from transformers import AutoTokenizer

CORPUS_PATH = WORK / "corpus" / "train_gold_v3.jsonl.gz"


def get_corpus_handle():
    if CORPUS_PATH.exists():
        return gzip.open(CORPUS_PATH, "rt", encoding="utf-8")
    for rid, fn, rtype in CORPUS_CANDIDATES:
        try:
            p = hf_hub_download(repo_id=rid, filename=fn, repo_type=rtype, token=HF_TOKEN,
                                local_dir=str(WORK / "corpus"))
            print("corpus source:", rid, rtype)
            return gzip.open(p, "rt", encoding="utf-8")
        except Exception as e:
            print("skip", rid, rtype, "->", type(e).__name__)
    raise RuntimeError("train_gold_v3.jsonl.gz not found in any candidate repo")


def deva_runs(text):
    runs, run = [], ""
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            run += ch
        elif run:
            if len(run) >= 2:
                runs.append(run)
            run = ""
    if len(run) >= 2:
        runs.append(run)
    return runs


def read_lines(n):
    lines = []
    with get_corpus_handle() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line)
    return lines


def ngram_counts(lines, max_n=6):
    cb = {n: Counter() for n in range(2, max_n + 1)}
    for line in lines:
        try:
            msgs = json.loads(line)["messages"]
        except Exception:
            continue
        for m in msgs:
            for run in deva_runs(m["content"]):
                for n in range(2, max_n + 1):
                    for i in range(len(run) - n + 1):
                        cb[n][run[i:i + n]] += 1
    return cb


def fertility(tok, texts):
    tc = sum(len(t) for t in texts)
    tt = sum(len(tok.encode(t)) for t in texts)
    return tt / tc * 1000 if tc else 0.0


def select_tokens(cb, budget):
    scored = []
    for n in range(2, 7):
        for ng, freq in cb[n].most_common(2000):
            ids = qtok.encode(ng)
            if len(ids) > 1:
                scored.append(((len(ids) - 1) * math.sqrt(freq), ng))
    scored.sort(key=lambda x: -x[0])
    chosen, seen = [], set()
    for _, ng in scored:
        if ng in seen:
            continue
        seen.add(ng)
        chosen.append(ng)
        if len(chosen) >= budget:
            break
    return chosen


if STATE.get("phase1"):
    TOKENS = STATE["tokens"]
    qtok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"phase1 done — {len(TOKENS)} tokens loaded")
else:
    t0 = time.time()
    qt = AutoTokenizer.from_pretrained(BASE_MODEL)
    qtok = qt
    mine_lines = read_lines(MINE_LINES)
    val_lines = []
    with get_corpus_handle() as f:
        for i, line in enumerate(f):
            if i < VAL_LINES_OFFSET:
                continue
            if i >= VAL_LINES_OFFSET + VAL_LINES_COUNT:
                break
            val_lines.append(line)

    val_deva = []
    for line in val_lines:
        try:
            for m in json.loads(line)["messages"]:
                val_deva.extend(deva_runs(m["content"]))
        except Exception:
            pass

    base_fert = fertility(qt, val_deva)
    print(f"baseline Devanagari fertility: {base_fert:.1f} tok/1000 chars "
          f"({len(val_deva)} runs, {time.time()-t0:.0f}s)")

    cb = ngram_counts(mine_lines)
    TOKENS = select_tokens(cb, BUDGET)
    print(f"selected {len(TOKENS)} tokens; top10:", TOKENS[:10])

    probe = AutoTokenizer.from_pretrained(BASE_MODEL)
    probe.add_tokens(TOKENS)
    new_fert = fertility(probe, val_deva)
    sanity_savings = (1 - new_fert / base_fert) * 100
    print(f"injected fertility: {new_fert:.1f} → savings {sanity_savings:.1f}%")

    STATE["phase1"] = True
    STATE["tokens"] = TOKENS
    STATE["fertility_base"] = round(base_fert, 1)
    STATE["fertility_injected"] = round(new_fert, 1)
    save_state(STATE)
