#@title 2 · Load v3 model + LoRA {display-mode:"form"}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

if not STATE.get("model_loaded"):
    t0 = time.time()
    print(f"loading {BASE_MODEL} ...", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN,
                                         trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    # freeze base, add LoRA
    for p in model.parameters():
        p.requires_grad = False

    lcfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.config.use_cache = False

    print(f"loaded in {time.time()-t0:.0f}s | {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")
    model.print_trainable_parameters()
    STATE["model_loaded"] = True
    save_state(STATE)
else:
    print("model already loaded")
