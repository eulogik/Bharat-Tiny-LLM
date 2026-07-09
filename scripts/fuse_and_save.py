"""
Fuse MLX-trained LoRA adapter into base model and save correctly.

Key fix: save state_dict DIRECTLY to safetensors (NOT model.save_pretrained)
which was corrupting weights (0.03-0.17 diffs between in-memory and saved).
"""

import torch, json, os, shutil, safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ADAPTER_REPO = "eulogik/Bharat-Tiny-LLM-adapter"
OUTPUT_DIR = "/tmp/bharat-tiny-llm-fused-v2"
DTYPE = torch.float16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {DEVICE}, dtype: {DTYPE}")

# ── 1. Load adapter weights ────────────────────────────────────────────
peft_path = hf_hub_download(ADAPTER_REPO, "adapter_model.safetensors")
peft = safetensors.torch.load_file(peft_path)
print(f"Loaded adapter: {len(peft)} tensors")

# ── 2. Load base model on MPS ──────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("Base model loaded")

# ── 3. Fuse LoRA weights ───────────────────────────────────────────────
# PEFT keys: base_model.model.model.layers.{N}.{module}.{proj}.lora_A.weight
# Model params: model.layers.{N}.{module}.{proj}.weight
scale = 16.0  # alpha / r
modified = 0
for name, param in model.named_parameters():
    if param.ndim != 2:
        continue
    layer_path = name.replace(".weight", "")
    lora_A_key = f"base_model.model.{layer_path}.lora_A.weight"
    lora_B_key = f"base_model.model.{layer_path}.lora_B.weight"
    if lora_A_key in peft and lora_B_key in peft:
        lora_A = peft[lora_A_key].to(device=DEVICE, dtype=DTYPE)
        lora_B = peft[lora_B_key].to(device=DEVICE, dtype=DTYPE)
        param.data += torch.mm(lora_B, lora_A) * scale
        modified += 1

print(f"Fused {modified} weight tensors")
assert modified == 112, f"Expected 112, got {modified}"

# ── 4. Save state_dict DIRECTLY (bypass save_pretrained) ───────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save model weights as safetensors in float16 (on CPU)
print("Saving fused state_dict...")
state_dict = {k: v.contiguous().to(dtype=DTYPE).cpu() for k, v in model.state_dict().items()}
safetensors.torch.save_file(state_dict, os.path.join(OUTPUT_DIR, "model.safetensors"))
print(f"Saved model.safetensors ({len(state_dict)} tensors)")

# ── 5. Copy config and tokenizer files from base model ─────────────────
print("Copying config + tokenizer files...")
from transformers import AutoConfig
config = AutoConfig.from_pretrained(BASE_MODEL)
config.torch_dtype = str(DTYPE).split(".")[-1]  # "float16"
config.save_pretrained(OUTPUT_DIR)
print("Saved config.json")

# Copy tokenizer files from base model
for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"]:
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bharat-tiny-llm-qwen-q4", fn) if False else None
    try:
        local_path = AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(OUTPUT_DIR)
    except:
        pass

# Simpler: just save tokenizer from the loaded one
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved tokenizer files")

# ── 6. Create generation config ────────────────────────────────────────
gen_config = {
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "pad_token_id": 151643,
}
with open(os.path.join(OUTPUT_DIR, "generation_config.json"), "w") as f:
    json.dump(gen_config, f, indent=2)

# ── 7. Verify ──────────────────────────────────────────────────────────
# Load back and compare
print("Verifying save/load roundtrip...")
restored = safetensors.torch.load_file(os.path.join(OUTPUT_DIR, "model.safetensors"))
max_diff = 0
for key in state_dict:
    diff = (state_dict[key] - restored[key]).abs().max().item()
    max_diff = max(max_diff, diff)
print(f"Max roundtrip diff: {max_diff:.10f} (should be 0.0)")
assert max_diff < 1e-6, f"Roundtrip failed: max diff {max_diff}"

# ── 8. Quick test inference ────────────────────────────────────────────
print("Quick inference test...")
test_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True
).to(DEVICE)
test_model.eval()

prompt = "<|im_start|>user\ntum chai peete ho?\n<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = test_model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Response: {repr(response)}")

print(f"\n✅ Fused model saved to {OUTPUT_DIR}")
print(f"Files: {os.listdir(OUTPUT_DIR)}")
