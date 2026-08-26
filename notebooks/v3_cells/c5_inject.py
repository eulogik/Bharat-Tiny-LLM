#@title 4 · PHASE 2 — Injection: tokenizer + mean-init embedding delta (~5 min) {display-mode:"form"}
import torch
from safetensors.torch import save_file, load_file

DELTA_PATH = WORK / "embed_delta.safetensors"
TOK_DIR = WORK / "v3_tokenizer"

if not STATE.get("phase2"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TOKENS = STATE["tokens"]
    pre_ids = {t: qtok.encode(t) for t in TOKENS}

    vtok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_len = len(vtok)
    added = vtok.add_tokens(TOKENS)
    assert added == len(TOKENS), f"added {added} != {len(TOKENS)}"
    new_ids = list(range(base_len, base_len + added))
    tok_to_id = {t: i for t, i in zip(TOKENS, new_ids)}

    print("loading base model on CPU for mean-init...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    emb = base_model.get_input_embeddings().weight.data
    rows = torch.zeros((len(TOKENS), emb.shape[1]), dtype=torch.float16)
    with torch.no_grad():
        for j, t in enumerate(TOKENS):
            const = torch.tensor(pre_ids[t], dtype=torch.long)
            rows[j] = emb[const].mean(dim=0).to(torch.float16)
    del base_model

    save_file({"ids": torch.tensor(new_ids, dtype=torch.int64),
               "rows": rows}, str(DELTA_PATH))
    vtok.save_pretrained(TOK_DIR)

    upload_file(path_or_fileobj=str(DELTA_PATH),
                path_in_repo="embed_delta.safetensors", repo_id=ARTIFACTS_REPO,
                repo_type="model", token=HF_TOKEN)
    for f in sorted(TOK_DIR.iterdir()):
        if f.is_file():
            upload_file(path_or_fileobj=str(f), path_in_repo=f"v3_tokenizer/{f.name}",
                        repo_id=ARTIFACTS_REPO, repo_type="model", token=HF_TOKEN)
    STATE["phase2"] = True
    save_state(STATE)
    print(f"delta saved: {rows.shape} fp16, ids {new_ids[0]}..{new_ids[-1]}")
else:
    if not DELTA_PATH.exists():
        hf_hub_download(repo_id=ARTIFACTS_REPO, filename="embed_delta.safetensors",
                        repo_type="model", token=HF_TOKEN, local_dir=str(WORK))
    if not TOK_DIR.exists():
        snapshot_download(repo_id=ARTIFACTS_REPO, repo_type="model", token=HF_TOKEN,
                          local_dir=str(WORK), allow_patterns=["v3_tokenizer/*"])
    print("phase2 artifacts fetched")

d = load_file(str(DELTA_PATH))
print("delta ready | new ids:", d["ids"].tolist()[:3], "... shape:", d["rows"].shape)
