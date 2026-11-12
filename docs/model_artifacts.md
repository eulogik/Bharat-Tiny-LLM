# Model artifacts and cleanup

**2026-11-12.** The rejected v4 fine-tuning checkpoints were deleted to free
~16.5 GB on the internal drive:

- `bharat-v4-merged`, `bharat-v4-tokenized`, `bharat-v4-warm`,
  `bharat-v4-base-merged`

The two LoRA adapters are archived on the external drive:

    /Volumes/KIOXIA 1TB/Brahmi_archive/v4_loras/bharat-v4-lora-base
    /Volumes/KIOXIA 1TB/Brahmi_archive/v4_loras/bharat-v4-lora-meaninit

They are research artifacts only. The shipped v4.0 system uses the stock
Qwen3.5-2B Q4_K_M GGUF, hosted at
https://huggingface.co/eulogik/Bharat-Tiny-LLM-v4.

The external drive cleanup removed only rejected GGUF test exports, duplicated
audio caches, HF caches, Rust build artifacts and superseded checkpoints. No
personal backups or source archives were touched.
