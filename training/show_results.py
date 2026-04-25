#!/usr/bin/env python3
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

base = "/mnt/c/Users/markj/Desktop/metaprelim/cyber_arena/training"

with open(f"{base}/pre_dpo_responses.json", encoding="utf-8") as f:
    pre = json.load(f)
with open(f"{base}/post_dpo_responses.json", encoding="utf-8") as f:
    post = json.load(f)

print("=" * 70)
print("BEHAVIORAL COMPARISON — Pre vs Post DPO")
print("=" * 70)

for scenario in pre:
    pre_r  = pre[scenario]
    post_r = post[scenario]
    print(f"\n{'='*20} {scenario} {'='*20}")
    print(f"\nPRE-DPO (500 chars):\n{pre_r[:500]}")
    print(f"\nPOST-DPO (500 chars):\n{post_r[:500]}")
    print()
    # Quick keyword scan
    keywords = ["http_get", "nmap", "execute_exploit", "probe", "honeypot",
                "avoid", "stealth", "detection", "SIEM", "lateral"]
    pre_hits  = [k for k in keywords if k.lower() in pre_r.lower()]
    post_hits = [k for k in keywords if k.lower() in post_r.lower()]
    print(f"  PRE keywords:  {pre_hits}")
    print(f"  POST keywords: {post_hits}")
    print("-" * 60)

# Show final DPO stats from last log entry
print("\n=== TRAINING METRICS (final steps) ===")
metrics = {
    "Final loss": "0.0633",
    "rewards/chosen (final)":   "+8.976",
    "rewards/rejected (final)": "-1.568",
    "rewards/margins (final)":  "+10.54",
    "rewards/accuracies":       "1.0 (100%)",
    "Runtime":                  "37m 53s",
}
for k, v in metrics.items():
    print(f"  {k:<30} {v}")
