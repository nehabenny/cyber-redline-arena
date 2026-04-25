#!/usr/bin/env python3
"""HuggingFace token verification and model access check via env var."""
import os, sys

# The HUGGING_FACE_HUB_TOKEN env var is read automatically by transformers/huggingface_hub
# without needing an interactive login() call.
TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("ERROR: HF_TOKEN env var not set")
    sys.exit(1)

# Set both env vars that HF hub checks
os.environ["HUGGING_FACE_HUB_TOKEN"] = TOKEN
os.environ["HF_TOKEN"] = TOKEN

print(f"=== HuggingFace Token Verification ===")
print(f"Token: {TOKEN[:8]}...{TOKEN[-4:]} (masked)")

from transformers import AutoTokenizer, AutoConfig
import warnings
warnings.filterwarnings("ignore")

print("\nStep 1/2: Downloading tokenizer...")
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-4B-Instruct",
    token=TOKEN,
    trust_remote_code=True
)
print(f"Tokenizer OK — vocab size: {tok.vocab_size}")

print("\nStep 2/2: Verifying config access...")
cfg = AutoConfig.from_pretrained(
    "Qwen/Qwen2.5-4B-Instruct",
    token=TOKEN,
    trust_remote_code=True
)
print(f"Config OK — model type: {cfg.model_type}")
print("\nModel accessible. Full weights will download when training starts.")
print("READY TO TRAIN")
