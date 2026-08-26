#@title 6a · PHASE 4 — Build capability SFT mix (~50K conversations) {display-mode:"form"}
SFT_DATA = WORK / "sft_data.jsonl.gz"

def norm_messages(rec):
    """Normalize common instruction schemas -> {'messages': [...]}. None if unusable."""
    try:
        if "messages" in rec and isinstance(rec["messages"], list):
            msgs = [{"role": m.get("role", "user"),
                     "content": str(m.get("content", ""))} for m in rec["messages"]]
        elif "instruction" in rec:
            user = rec["instruction"] + ("\n" + rec["input"] if rec.get("input") else "")
            msgs = [{"role": "user", "content": str(user)},
                    {"role": "assistant", "content": str(rec.get("output", ""))}]
        elif "question" in rec and "answer" in rec:
            msgs = [{"role": "user", "content": str(rec["question"])},
                    {"role": "assistant", "content": str(rec["answer"])}]
        elif "prompt" in rec and "response" in rec:
            msgs = [{"role": "user", "content": str(rec["prompt"])},
                    {"role": "assistant", "content": str(rec["response"])}]
        else:
            return None
    except Exception:
        return None
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
        if all(len(m["content"].strip()) > 1 for m in msgs):
            return {"messages": msgs}
    return None


def gsm_answer_fix(ans):
    import re as _re
    body = _re.sub(r"<<[^>]*>>", "", ans)
    m = _re.search(r"#{3,4}\s*(-?[\d,]+(?:\.\d+)?)", body)
    if m:
        final = m.group(1).replace(",", "")
        body = body.split("####")[0].split("###")[0].rstrip()
        return body.strip() + f"\nउत्तर: {final}"
    return body.strip()


if STATE.get("sft_data") and SFT_DATA.exists():
    print("sft_data ready")
else:
    from datasets import load_dataset
    random.seed(42)
    out, seen = [], set()

    def add(msgs, src):
        key = hash(json.dumps(msgs)[:400])
        if key not in seen:
            seen.add(key)
            out.append(msgs)

    # 1) GSM8K-Hindi train — teaches math CoT + 'उत्तर:' marker format
    gsm = load_dataset("bingbangboom/gsm8k-hindi")["train_main"]
    for r in gsm:
        ans = gsm_answer_fix(r["answer"])
        add([{"role": "user", "content": r["question"].strip()},
             {"role": "assistant", "content": ans}], "gsm")
    print("after gsm:", len(out))

    # 2) IndicVault (Hindi+Hinglish QA) — knowledge breadth, streamed subset
    try:
        iv = load_dataset("maya-research/IndicVault", split="train", streaming=True)
        want = min(30000, MAX_SFT_ROWS - len(out))
        got = 0
        for r in iv:
            if got >= want or len(out) >= MAX_SFT_ROWS:
                break
            m = norm_messages(r)
            if m:
                add(m, "iv")
                got += 1
    except Exception as e:
        print("IndicVault skipped:", type(e).__name__, e)
    print("after indicvault:", len(out))

    # 3) Hinglish instruct + govt schemes + recipes
    for rid, split in [("Sujalvc/hinglish-instruct-dataset", "train"),
                       ("Subh24ai/yojana-sahayak-instruct", "train"),
                       ("VishalMysore/cookGPT", "train")]:
        if len(out) >= MAX_SFT_ROWS:
            break
        try:
            dsi = load_dataset(rid, split=split)
            for r in dsi:
                if len(out) >= MAX_SFT_ROWS:
                    break
                m = norm_messages(r)
                if m:
                    add(m, rid)
        except Exception as e:
            print(rid, "skipped:", type(e).__name__)
    print("total:", len(out))

    random.shuffle(out)
    import gzip as _gz
    with _gz.open(SFT_DATA, "wt", encoding="utf-8") as f:
        for m in out:
            f.write(json.dumps({"messages": m["messages"]}) + "\n")
    upload_file(str(SFT_DATA), "sft_data.jsonl.gz", ARTIFACTS_REPO,
                repo_type="model", token=HF_TOKEN)
    STATE["sft_data"] = True
    STATE["sft_rows"] = len(out)
    save_state(STATE)

#@title 6b · PHASE 4 — Self-healing SFT runner on 2×T4 {display-mode:"form"}
LOG_PATH = WORK / "sft_train.log"
HEARTBEAT = WORK / "progress.json"
DEADLINE = time.time() + MAX_SESSION_HOURS * 3600


def tail(path, n=1800):
    try:
        return Path(path).read_text(errors="ignore")[-n:]
    except Exception:
        return ""


def heartbeat_step():
    try:
        return json.loads(HEARTBEAT.read_text()).get("step", -1)
    except Exception:
        return -1


if not STATE.get("sft_done"):
    resume_dir = fetch_latest_sft_checkpoint(WORK / "resume")
    prev_best = heartbeat_step() if HEARTBEAT.exists() else -1
    bs, accum = SFT_PER_DEVICE_BS, SFT_GRAD_ACCUM
    stall = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if time.time() > DEADLINE:
            kill_session(f"session deadline hit before attempt {attempt}")
        print(f"\n=== SFT attempt {attempt}/{MAX_ATTEMPTS} | bs={bs} accum={accum} "
              f"| resume={resume_dir} ===", flush=True)

        cmd = ["torchrun", "--nproc_per_node=2", "--master_port=29515",
               "train_sft.py",
               "--data", str(SFT_DATA), "--out", str(WORK / "sft_out"),
               "--tok-dir", str(TOK_DIR), "--delta", str(DELTA_PATH),
               "--warmup-rows", str(WARMUP_PATH),
               "--bs", str(bs), "--accum", str(accum),
               "--epochs", str(SFT_EPOCHS), "--lr", str(SFT_LR),
               "--seq", str(SFT_SEQ)]
        if resume_dir:
            cmd += ["--resume-dir", str(resume_dir)]

        env = dict(os.environ, BASE_MODEL=BASE_MODEL, HF_TOKEN=HF_TOKEN)
        with open(LOG_PATH, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=str(WORK), env=env)
            start_best = max(prev_best, 0)
            while proc.poll() is None:
                time.sleep(30)
                hs = heartbeat_step()
                if hs > prev_best:
                    prev_best = hs
                    print(f"  [hb] step={hs}", flush=True)
                if time.time() > DEADLINE:
                    print("deadline exceeded mid-run; terminating")
                    proc.terminate()
                    proc.wait(120)
                    break

        rc = proc.poll()
        done_flag = (WORK / "sft_out" / "DONE").exists()
        logtail = tail(LOG_PATH)

        if rc == 0 and done_flag:
            print("SFT finished successfully")
            break

        oom = ("OutOfMemoryError" in logtail) or ("CUDA out of memory" in logtail)
        progressed = prev_best > start_best
        stall = stall + (0 if progressed else 1)

        print(f"attempt {attempt}: rc={rc} done={done_flag} "
              f"oom={oom} progressed={progressed} stall={stall}")

        if oom and bs > MIN_PER_DEVICE_BS:
            bs = max(MIN_PER_DEVICE_BS, bs // 2)
            accum *= 2
            print(f"OOM -> shrinking to bs={bs}, accum={accum}")
        elif not progressed:
            print(f"NO step progress this attempt ({stall}/{STALL_LIMIT})")

        resume_dir = fetch_latest_sft_checkpoint(WORK / "resume")

        if stall >= STALL_LIMIT:
            kill_session(f"training stalled: {stall} attempts with no progress")
    else:
        kill_session(f"max attempts ({MAX_ATTEMPTS}) exhausted")

    final_dir = WORK / "sft_out" / "final_adapter"
    assert final_dir.exists(), "no final adapter after successful run"
    upload_folder(folder_path=str(final_dir), repo_id=ARTIFACTS_REPO,
                  path_in_repo="sft/final", repo_type="model", token=HF_TOKEN)
    tokdir = WORK / "sft_out" / "tokenizer"
    for f in tokdir.iterdir():
        if f.is_file():
            upload_file(str(f), f"sft/final_tokenizer/{f.name}", ARTIFACTS_REPO,
                        repo_type="model", token=HF_TOKEN)
    STATE["sft_done"] = True
    save_state(STATE)
else:
    print("SFT already done")
