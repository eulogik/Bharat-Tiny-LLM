#@title 2 · State manager — persists to HF Hub, survives any session death {display-mode:"form"}
STATE_PATH = WORK / "state_v3.json"


def load_state():
    try:
        p = hf_hub_download(repo_id=ARTIFACTS_REPO, filename="state_v3.json", repo_type="model",
                            token=HF_TOKEN, local_dir=str(WORK), force_download=True)
        st = json.loads(Path(p).read_text())
        print("loaded remote state")
        return st
    except Exception as e:
        print(f"fresh run (no remote state: {type(e).__name__})")
        return {}


def save_state(st):
    STATE_PATH.write_text(json.dumps(st, indent=2))
    upload_file(path_or_fileobj=str(STATE_PATH), path_in_repo="state_v3.json",
                repo_id=ARTIFACTS_REPO, repo_type="model", token=HF_TOKEN,
                commit_message="state update")
    flags = {k: v for k, v in st.items() if isinstance(v, (int, str, bool))}
    print("state saved:", flags)


def ensure_repos():
    create_repo(ARTIFACTS_REPO, private=True, exist_ok=True, token=HF_TOKEN)
    create_repo(OUT_REPO, private=True, exist_ok=True, token=HF_TOKEN)


def fetch_latest_sft_checkpoint(local_dir):
    """Download newest sft/checkpoint-N adapter from Hub. Returns local path or None."""
    try:
        files = list_repo_files(ARTIFACTS_REPO, repo_type="model", token=HF_TOKEN)
    except Exception:
        return None
    steps = []
    for f in files:
        m = re.match(r"sft/checkpoint-(\d+)/adapter_model\.safetensors$", f)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        return None
    latest = max(steps)
    target = Path(local_dir) / f"sft/checkpoint-{latest}"
    if target.exists():
        return target
    snapshot_download(ARTIFACTS_REPO, repo_type="model", token=HF_TOKEN,
                      local_dir=local_dir,
                      allow_patterns=[f"sft/checkpoint-{latest}/**"])
    if target.exists():
        print(f"fetched remote checkpoint step {latest}")
        return target
    return None


STATE = load_state()
ensure_repos()
print("phase status:", {k: STATE.get(k) for k in
      ("phase1", "phase2", "phase3", "sft_data", "sft_done", "phase5")})
