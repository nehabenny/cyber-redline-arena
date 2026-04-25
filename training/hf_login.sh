#!/bin/bash
# HuggingFace login + download model weights
# Usage: wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/path/training/hf_login.sh YOUR_HF_TOKEN

echo "=== HuggingFace Login + Model Pre-Download ==="

TOKEN=${1:-""}
if [ -z "$TOKEN" ]; then
    echo "Usage: bash hf_login.sh YOUR_HF_TOKEN"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "(Read access is free — no payment needed)"
    exit 1
fi

# Install huggingface_hub
pip3 install -q huggingface_hub

# Login
python3 -c "
from huggingface_hub import login
login(token='$TOKEN', add_to_git_credential=False)
print('Logged in to HuggingFace successfully')
"

echo ""
echo "=== Pre-downloading Qwen/Qwen2.5-4B-Instruct ==="
echo "This will take 5-10 minutes (8.1GB model)..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-4B-Instruct', trust_remote_code=True)
print('Tokenizer OK')

print('Downloading model (this is the large download -- grab a coffee)...')
# Just download the config to verify access -- full model loads during training
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-4B-Instruct', ignore_patterns=['*.ot', '*.msgpack'])
print('Model cached successfully -- ready to train!')
"
