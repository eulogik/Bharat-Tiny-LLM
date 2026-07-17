"""
Generate test samples from a fused model to evaluate real quality.
Usage: python3 scripts/gen_eval.py <model_dir> <label>
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_best0937"
label = sys.argv[2] if len(sys.argv) > 2 else model_dir
DTYPE = torch.float16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

prompts = [
    "tum chai peete ho?",
    "Mujhe ek chutti ka plan batao.",
    "भारत के कुछ प्रसिद्ध त्योहार बताओ।",
    "Kal mera interview hai, kuch tips do.",
    "Ek choti si kahani likho jisme ek billi ho.",
    "Cooking mein garam masale kyu zaroori hain?",
]

print(f"\n{'='*60}\nEVAL: {label}\n{'='*60}")
for p in prompts:
    chat = [{"role": "user", "content": p}]
    prompt_txt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_txt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=120, temperature=0.8, do_sample=True,
            top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nUSER: {p}")
    print(f"BOT:  {resp.strip()}")
