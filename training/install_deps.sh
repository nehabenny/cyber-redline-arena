#!/bin/bash
# Install all DPO training dependencies in WSL2
# Run as: wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/Users/markj/Desktop/metaprelim/cyber_arena/training/install_deps.sh

set -e
echo "=== Installing DPO training dependencies ==="

pip3 install --quiet numpy
pip3 install --quiet transformers accelerate bitsandbytes peft trl datasets
pip3 install --quiet openenv gymnasium openai
pip3 install --quiet matplotlib

echo ""
echo "=== Verifying installs ==="
python3 -c "
import torch
import transformers
import trl
import peft
import datasets
import bitsandbytes
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('trl:', trl.__version__)
print('peft:', peft.__version__)
print('CUDA available:', torch.cuda.is_available())
print('ALL OK')
"
