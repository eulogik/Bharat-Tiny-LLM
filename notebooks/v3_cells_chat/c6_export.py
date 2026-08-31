#@title 5 · Merge adapter + upload v3.1 {display-mode:"form"}
if not STATE.get("export_done"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import upload_folder, upload_file

    t0 = time.time()
    final_dir = OUT_DIR / "final_adapter"
    assert final_dir.exists(), f"no adapter at {final_dir}"

    print("loading base v3 fp32 for merge...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32)

    model = PeftModel.from_pretrained(base, str(final_dir))
    model = model.merge_and_unload()
    model.half()

    export = WORK / "export_v31"
    model.save_pretrained(str(export), safe_serialization=True,
                          max_shard_size="4GB")
    tok_out = AutoTokenizer.from_pretrained(str(OUT_DIR / "tokenizer"))
    tok_out.save_pretrained(str(export))

    print("uploading to hub...", flush=True)
    upload_folder(folder_path=str(export), repo_id=OUT_REPO,
                  repo_type="model", token=HF_TOKEN,
                  commit_message="v3.1: conversational SFT on samvaad+hinglish+corpus")

    STATE["export_done"] = True
    STATE["export_time_min"] = round((time.time() - t0) / 60, 1)
    save_state(STATE)
    print(f"v3.1 exported in {time.time()-t0:.0f}s → {OUT_REPO}")
else:
    print("export already done")

# ── summary ──────────────────────────────────────────────────────
s = load_state()
print(f"""
=== Bharat-Tiny-LLM v3.1 Chat SFT Complete ===
  model:   {OUT_REPO}
  data:    {CHAT_DATA} ({s.get('train_rows','?')} train / {s.get('eval_rows','?')} eval)
  lora:    r={LORA_R} alpha={LORA_ALPHA}
  time:    train {s.get('train_time_min','?')} min | export {s.get('export_time_min','?')} min
  next:    run benchmarks locally (fertility, PPL, GSM8K-Hindi, chat eval)
""")
