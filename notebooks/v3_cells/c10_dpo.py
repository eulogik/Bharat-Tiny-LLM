#@title 8 · OPTIONAL — DPO polish (RUN_DPO flag in CONFIG; default off) {display-mode:"form"}
if not RUN_DPO:
    print("RUN_DPO=False -> skipping. Enable after SFT quality review.")
else:
    # Expects dpo_pairs.jsonl.gz on the artifacts repo:
    #   {"prompt": [...messages...], "chosen": "...", "rejected": "..."}
    %pip install -q trl 2>&1 | tail -1
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file

    hf_hub_download(repo_id=ARTIFACTS_REPO, filename="dpo_pairs.jsonl.gz",
                    repo_type="model", token=HF_TOKEN, local_dir=str(WORK))
    pairs = load_dataset("json", data_files=str(WORK / "dpo_pairs.jsonl.gz"))["train"]

    import torch
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).cuda()
    base_vocab = model.get_input_embeddings().num_embeddings
    d = load_file(str(DELTA_PATH))
    ids = d["ids"].tolist()
    model.resize_token_embeddings(base_vocab + len(ids))
    with torch.no_grad():
        model.get_input_embeddings().weight[ids] = d["rows"].to(torch.float16)
    tokd = AutoTokenizer.from_pretrained(TOK_DIR)

    cfg = DPOConfig(
        output_dir=str(WORK / "dpo_out"), per_device_train_batch_size=2,
        gradient_accumulation_steps=8, beta=0.1, learning_rate=5e-7,
        num_train_epochs=1, logging_steps=25, save_steps=500,
        fp16=True, report_to=[],
    )
    dtrainer = DPOTrainer(model=model, args=cfg,
                          train_dataset=pairs, processing_class=tokd)
    dtrainer.train()
    dtrainer.save_model(str(WORK / "dpo_final"))
    upload_folder(folder_path=str(WORK / "dpo_final"), repo_id=ARTIFACTS_REPO,
                  path_in_repo="dpo/final", repo_type="model", token=HF_TOKEN)
    print("DPO done -> re-run PHASE 5 to export merged model")
