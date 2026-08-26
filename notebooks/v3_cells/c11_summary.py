#@title 9 · Final status {display-mode:"form"}
import json, time
print(json.dumps({k: v for k, v in STATE.items() if k != "tokens"},
      indent=2, default=str))
print(f"\ntokens injected: {len(STATE.get('tokens', []))}")
print(f"artifacts repo : https://huggingface.co/{ARTIFACTS_REPO} (private)")
print(f"model repo     : https://huggingface.co/{OUT_REPO} (private until release)")
elapsed_h = 0
print("\nIf the session died mid-way: just Runtime > Run all. "
      "Completed phases are skipped automatically.")
