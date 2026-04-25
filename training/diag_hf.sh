#!/bin/bash
# Test HuggingFace connectivity and diagnose access issues
echo "=== Network Connectivity Test ==="

echo "1. Ping huggingface.co:"
ping -c 2 huggingface.co 2>&1 | tail -3

echo ""
echo "2. HTTP GET test (curl):"
curl -s -o /dev/null -w "HTTP status: %{http_code}\n" "https://huggingface.co" 2>&1

echo ""
echo "3. HF API direct access (no auth):"
curl -s -w "\nHTTP: %{http_code}\n" "https://huggingface.co/api/models/Qwen/Qwen2.5-4B-Instruct" 2>&1 | tail -5

echo ""
echo "4. HF API with token:"
curl -s -w "\nHTTP: %{http_code}\n" \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/models/Qwen/Qwen2.5-4B-Instruct" 2>&1 | python3 -c "
import sys, json
raw = sys.stdin.read()
lines = raw.strip().split('\n')
# Print last line (HTTP code)
print(lines[-1])
# Try parse JSON
try:
    data = json.loads('\n'.join(lines[:-1]))
    print('Model ID:', data.get('id', 'N/A'))
    print('Private:', data.get('private', 'N/A'))
    print('Access OK')
except Exception as e:
    print('Parse error:', e)
    print('Raw:', raw[:300])
"
