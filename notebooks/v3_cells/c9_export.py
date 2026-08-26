#@title 7 · PHASE 5 — Merge LoRA + upload Bharat-Tiny-LLM-v3 (~10 min) {display-mode:"form"}
if not STATE.get("phase5"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file

    print("loading base fp32 on CPU for exact merge...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    base_vocab = model.get_input_embeddings().num_embeddings

    d = load_file(str(DELTA_PATH))
    ids = d["ids"].tolist()
    assert ids == list(range(base_vocab, base_vocab + len(ids)))
    model.resize_token_embeddings(base_vocab + len(ids))
    emb = model.get_input_embeddings()
    with torch.no_grad():
        emb.weight[ids] = d["rows"].float()
        if WARMUP_PATH.exists():
            w = load_file(str(WARMUP_PATH))
            emb.weight[w["ids"].tolist()] = w["rows"].float()
            print("warmup rows merged")

    model = PeftModel.from_pretrained(model, str(WORK / "sft_out" / "final_adapter"))
    model = model.merge_and_unload()
    model.half()

    EXPORT_DIR = WORK / "export"
    model.save_pretrained(str(EXPORT_DIR), safe_serialization=True,
                          max_shard_size="4GB")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    tok.save_pretrained(str(EXPORT_DIR))

    upload_folder(folder_path=str(EXPORT_DIR), repo_id=OUT_REPO,
                  repo_type="model", token=HF_TOKEN,
                  commit_message=f"v3: Qwen3-1.7B + Brahmi-{BUDGET} + capability SFT")
    STATE["phase5"] = True
    save_state(STATE)
    del model
else:
    print("phase5 done — model live at:", OUT_REPO)

print(f"""
=== M4-side next steps (MLX quantize + benchmarks; cannot run on T4) ===

# 1. quantize to 4-bit MLX (on the Mac):
python -m mlx_lm convert --hf-path {OUT_REPO} \\
    -q --q-bits 4 --q-group-size 64 \\
    --mlx-path "/Volumes/KIOXIA 1TB/brahmi-v3-q4"

# 2. point benchmarks at v3 (edit V2_MODEL/V2_ADAPTER constants):
#    scripts/bench_fertility.py   -> fertility vs Qwen3 tokenizer
#    scripts/bench_ppl.py         -> bits/char fair comparison
#    scripts/bench_gsm_hi.py      -> does SFT fix the 0% math score?
""")
