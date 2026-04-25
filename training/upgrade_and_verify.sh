#!/bin/bash
# Upgrade packages and verify versions
pip3 install -q --upgrade transformers huggingface_hub tokenizers 2>&1 | tail -3

python3 - <<'EOF'
import transformers, huggingface_hub
print("transformers:", transformers.__version__)
print("hf_hub:", huggingface_hub.__version__)

# Test HF connectivity with token via env var
import os
TOKEN = os.environ.get("HF_TOKEN", "")
print("Token set:", bool(TOKEN))

# Check if Qwen2.5-4B-Instruct is accessible
from huggingface_hub import HfApi
api = HfApi(token=TOKEN)
try:
    info = api.model_info("Qwen/Qwen2.5-4B-Instruct")
    print("Model accessible:", info.modelId)
    print("READY")
except Exception as e:
    print("Model access error:", e)
    # Try alternative names
    for name in ["Qwen/Qwen2.5-4B-Instruct", "Qwen/Qwen2.5-4B"]:
        try:
            info = api.model_info(name)
            print("Found model:", info.modelId)
            break
        except:
            print("Not found:", name)
EOF
