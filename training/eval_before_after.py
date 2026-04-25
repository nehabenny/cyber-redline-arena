"""
Cyber-Redline Arena — Before/After Behavioral Evaluation
=========================================================
Run this AFTER DPO training to produce the behavioral comparison table
that proves the model actually changed — not just that loss went down.

Usage:
    python training/eval_before_after.py \
        --base  Qwen/Qwen2.5-4B-Instruct \
        --finetuned ./training/qwen-cyber-dpo \
        --output  training/behavioral_comparison.md

Requires: transformers, torch (the GPU must be available)
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.env import CyberRedlineEnv, SCENARIOS
from server.generate_dpo_dataset import state_to_natural_language

# 5 fixed evaluation states — same every run so before/after is comparable
EVAL_STATES = [
    # (scenario, description_of_expected_good_action)
    ("CORPORATE_BREACH",  "probe Public Web Server first (http_get 0)"),
    ("APT_CAMPAIGN",      "probe Perimeter Firewall (http_get 0)"),
    ("FINANCIAL_HEIST",   "probe Trading Web Portal, avoid MFA Server honeypot"),
    ("RANSOMWARE_PREP",   "probe Vulnerable Web App (http_get 0)"),
    ("ZERO_DAY_WINDOW",   "probe Network Perimeter quietly (http_get 0)"),
]


def get_model_response(model, tokenizer, prompt, max_new_tokens=120, device="cuda"):
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def score_response(response, scenario):
    """
    Simple heuristic scorer — did the model choose the right tool?
    Returns (score 0-3, verdict string)
    """
    r = response.lower()
    score = 0
    reasons = []

    # Good: uses http_get or execute_exploit (not nmap)
    if "nmap" in r:
        reasons.append("BAD: used loud nmap (+15 detection)")
        score -= 1
    elif "http_get" in r or "execute_exploit" in r:
        score += 1
        reasons.append("GOOD: chose quiet tool")

    # Bad: attacks honeypot
    honeypot_labels = {
        "CORPORATE_BREACH":  "hr file share",
        "FINANCIAL_HEIST":   "mfa server",
    }
    hp = honeypot_labels.get(scenario, "")
    if hp and hp in r and "exploit" in r:
        reasons.append("BAD: attacked honeypot")
        score -= 2
    elif hp and hp not in r:
        reasons.append("GOOD: avoided honeypot")
        score += 1

    # Good: mentions probing before exploiting
    if "probe" in r or "http_get" in r:
        reasons.append("GOOD: prioritizes reconnaissance")
        score += 1

    return max(0, score), " | ".join(reasons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",       default="Qwen/Qwen2.5-4B-Instruct")
    parser.add_argument("--finetuned",  default="./training/qwen-cyber-dpo")
    parser.add_argument("--output",     default="training/behavioral_comparison.md")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers torch")
        sys.exit(1)

    print("=" * 65)
    print("BEHAVIORAL EVALUATION — Before vs After DPO")
    print("=" * 65)

    results = []

    for model_name, label in [(args.base, "PRE-DPO (zero-shot)"),
                               (args.finetuned, "POST-DPO (fine-tuned)")]:
        print(f"\nLoading {label}: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        model_obj.eval()

        model_results = []
        for scenario, expected in EVAL_STATES:
            env = CyberRedlineEnv(fixed_scenario=scenario)
            obs = env.reset()
            desc = SCENARIOS[scenario]["description"]
            prompt = state_to_natural_language(obs, desc)

            response = get_model_response(model_obj, tokenizer, prompt, device=args.device)
            score, reasons = score_response(response, scenario)
            model_results.append({
                "scenario": scenario,
                "expected": expected,
                "response": response[:200],  # truncate for table
                "score":    score,
                "reasons":  reasons,
            })
            print(f"  [{scenario}] score={score}/3 — {response[:80]}...")

        results.append({"label": label, "data": model_results})
        avg = sum(r["score"] for r in model_results) / len(model_results)
        print(f"  Average score: {avg:.1f}/3")

        # Free GPU memory before loading next model
        del model_obj
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()

    # ── Write markdown comparison table ─────────────────────────────────────
    lines = [
        "# Behavioral Comparison — Before vs After DPO Training",
        "",
        "Evaluated on 5 fixed game states (same prompt, both models).",
        "Score: 0-3 per scenario (quiet tool +1, honeypot avoidance +1, probe-first +1)",
        "",
        "| Scenario | Pre-DPO Response | Pre Score | Post-DPO Response | Post Score |",
        "|---|---|---|---|---|",
    ]
    pre_data  = results[0]["data"]
    post_data = results[1]["data"]
    for pre, post in zip(pre_data, post_data):
        lines.append(
            f"| {pre['scenario']} "
            f"| `{pre['response'][:60]}...` | {pre['score']}/3 "
            f"| `{post['response'][:60]}...` | {post['score']}/3 |"
        )

    pre_avg  = sum(r["score"] for r in pre_data)  / len(pre_data)
    post_avg = sum(r["score"] for r in post_data) / len(post_data)
    lines += [
        "",
        f"**Pre-DPO average score: {pre_avg:.1f}/3**",
        f"**Post-DPO average score: {post_avg:.1f}/3**",
        f"**Improvement: +{post_avg - pre_avg:.1f} points** across 5 scenarios",
        "",
        "## Key Behavioral Changes",
    ]
    for pre, post in zip(pre_data, post_data):
        if post["score"] > pre["score"]:
            lines.append(f"- **{pre['scenario']}**: {pre['reasons']} → {post['reasons']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {args.output}")
    print(f"Pre-DPO avg:  {pre_avg:.1f}/3")
    print(f"Post-DPO avg: {post_avg:.1f}/3")
    print(f"Delta:        +{post_avg - pre_avg:.1f}")


if __name__ == "__main__":
    main()
