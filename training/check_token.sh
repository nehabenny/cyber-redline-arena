#!/bin/bash
# Find accessible model alternatives and check token permissions
TOKEN="${HF_TOKEN}"

echo "=== Token Info ==="
curl -s -H "Authorization: Bearer $TOKEN" "https://huggingface.co/api/whoami" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Username:', d.get('name','?'))
print('Email:', d.get('email','?'))
print('Type:', d.get('type','?'))
"

echo ""
echo "=== Testing model access ==="
# Test several public Qwen models
for MODEL in "Qwen/Qwen2.5-4B-Instruct" "Qwen/Qwen2-1.5B" "Qwen/Qwen2-1.5B-Instruct" "microsoft/phi-2" "HuggingFaceTB/SmolLM-1.7B-Instruct"; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "https://huggingface.co/api/models/$MODEL")
    echo "  $MODEL -> HTTP $CODE"
done
