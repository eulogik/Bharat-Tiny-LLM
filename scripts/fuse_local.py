"""
Fuse MLX-trained LoRA adapter (local safetensors) into Qwen2.5-1.5B base.

MLX LoRA keys: model.layers.{N}.{module}.{proj}.lora_a / .lora_b
Base param:     model.layers.{N}.{module}.{proj}.weight
Fusion:         W += (lora_b @ lora_a) * scale
"""
import sys, os, json, torch, safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DTYPE = torch.float16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SCALE = 2.0  # alpha/r = 32/16

adapter_path = sys.argv[1] if len(sys.argv) > 1 else "models/adapters/qwen_v2/0010000_adapters.safetensors"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fused_eval"

print(f"Device: {DEVICE} | dtype: {DTYPE} | scale: {SCALE}")
print(f"Adapter: {adapter_path}")

peft = safetensors.torch.load_file(adapter_path)
print(f"Loaded adapter: {len(peft)} tensors")

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("Base model loaded")

modified = 0
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        layer_path = name.replace(".weight", "")
        a_key = f"{layer_path}.lora_a"
        b_key = f"{layer_path}.lora_b"
        if a_key in peft and b_key in peft:
            lora_a = peft[a_key].to(device=DEVICE, dtype=DTYPE)
            lora_b = peft[b_key].to(device=DEVICE, dtype=DTYPE)
            # MLX: lora_a (in, r), lora_b (r, out) → delta (out, in)
            delta = torch.mm(lora_a, lora_b) * SCALE
            param.data += delta.T
            modified += 1

print(f"Fused {modified} weight tensors (expected 112)")
assert modified == 112, f"Expected 112, got {modified}"

os.makedirs(out_dir, exist_ok=True)
state_dict = {k: v.contiguous().to(dtype=DTYPE).cpu() for k, v in model.state_dict().items()}
safetensors.torch.save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
print(f"Saved model.safetensors ({len(state_dict)} tensors)")

config = AutoConfig.from_pretrained(BASE_MODEL)
config.torch_dtype = "float16"
config.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

gen_config = {"bos_token_id": 151643, "eos_token_id": 151643, "pad_token_id": 151643}
with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
    json.dump(gen_config, f, indent=2)

# Verify roundtrip
restored = safetensors.torch.load_file(os.path.join(out_dir, "model.safetensors"))
max_diff = max((state_dict[k] - restored[k]).abs().max().item() for k in state_dict)
print(f"Max roundtrip diff: {max_diff:.10f}")
assert max_diff < 1e-6

print(f"✅ Fused model saved to {out_dir}")
