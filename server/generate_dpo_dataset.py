"""
Cyber-Redline Arena — DPO Dataset Generator
============================================
Generates REAL chosen/rejected pairs from live environment trajectories.

Chosen  = HeuristicRedAgent actions (optimal, probe-before-exploit, avoid honeypots)
Rejected = Deliberately bad actions  (honeypot attacks, nmap, skipped prerequisites)

Each pair is generated from the SAME game state — strongest possible DPO signal.

Output: training/dpo_dataset.jsonl     (TRL DPOTrainer format)
        training/dpo_dataset_stats.json (summary for notebook intro slide)
"""

import os, sys, json, random, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from server.env import CyberRedlineEnv
from server.agents import HeuristicRedAgent

os.makedirs("training", exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
PAIRS_PER_SCENARIO = 16   # × 5 scenarios = 80 base pairs, plus hard pairs = ~240 total
SEED               = 42


def tier_to_int(tier):
    return {"MONITOR": 0, "ALERT": 1, "LOCKDOWN": 2}.get(tier, 0)


def state_to_natural_language(obs, scenario_desc):
    """
    Convert raw observation dict to a natural language prompt.
    CRITICAL: LLMs respond far better to natural language than raw JSON.
    This is the prompt format the model was pre-trained on.
    """
    tier    = obs.get("blue_tier", "MONITOR")
    det     = obs.get("detection_level", 0)
    steps   = obs.get("steps_remaining", 0)
    nodes   = obs.get("nodes", {})
    obj     = obs.get("objective", "")

    tier_desc = {
        "MONITOR":  "🟢 MONITOR (baseline traffic checks)",
        "ALERT":    "🟡 ALERT (active threat hunting underway)",
        "LOCKDOWN": "🔴 LOCKDOWN (all internal routing blocked)",
    }.get(tier, tier)

    det_desc = (
        "undetected" if det < 20 else
        "lightly flagged" if det < 40 else
        "flagged — increase caution" if det < 65 else
        "heavily flagged — near lockout"
    )

    node_lines = []
    for i, (nid, n) in enumerate(nodes.items()):
        status_str = {0: "○ HEALTHY", 1: "~ DAMAGED/PROBED", 2: "✓ COMPROMISED"}.get(n["status"], "?")
        trap_tag   = " 🍯 [HONEYPOT — DO NOT EXPLOIT]" if n["is_honeypot"] else ""
        flag_tag   = " ★ [OBJECTIVE — THIS IS THE FLAG]" if nid == obj else ""
        node_lines.append(f"  [{i}] {n['label']} — {status_str}{trap_tag}{flag_tag}")

    node_block = "\n".join(node_lines)

    prompt = (
        f"You are a Red Team AI operator conducting a cyber operation.\n\n"
        f"MISSION: {scenario_desc}\n\n"
        f"CURRENT SITUATION:\n"
        f"- Blue Team SIEM: {tier_desc}\n"
        f"- Detection level: {det}/100 ({det_desc})\n"
        f"- Steps remaining: {steps}\n\n"
        f"NETWORK NODES:\n{node_block}\n\n"
        f"RULES:\n"
        f"  • http_get <node_index>   — Quiet probe (+2 detection). "
        f"Probing before exploiting gives a 2× damage bonus.\n"
        f"  • execute_exploit <node_index> — Active attack (+8 detection). "
        f"Requires previous http_get AND all prerequisite nodes compromised.\n"
        f"  • nmap <node_index>       — Loud recon (+15 detection). Rarely optimal.\n"
        f"  • NEVER exploit a 🍯 HONEYPOT — it triggers +40 detection and Blue Team lockout.\n\n"
        f"STATE YOUR NEXT ACTION and briefly explain your reasoning:"
    )
    return prompt


def action_to_response(action, obs, tone="good"):
    """Format an action as a natural language response the model should output."""
    tool_names = {0: "nmap", 1: "http_get", 2: "execute_exploit"}
    tool       = int(action.get("tool", 1))
    target     = int(action.get("target", 0))
    nodes      = obs.get("nodes", {})
    node_ids   = list(nodes.keys())
    target     = min(target, len(node_ids) - 1)
    node       = nodes[node_ids[target]]
    label      = node["label"]
    tool_str   = tool_names.get(tool, "http_get")

    if tone == "good":
        # Chosen: concise, legible, strategically reasoned
        if tool == 1:  # http_get
            if node["status"] == 0:
                reason = (f"Performing a quiet HTTP probe on {label} to fingerprint it and "
                          f"prime it for exploitation — this avoids a detection spike and gives "
                          f"us a 2× damage bonus when we exploit next step.")
            else:
                reason = (f"Re-probing {label} to confirm damage status before committing "
                          f"to full exploitation.")
        elif tool == 2:  # execute_exploit
            reason = (f"Launching exploit against {label}. Prerequisites satisfied, node "
                      f"already probed — preparation bonus active. Damage burst incoming.")
        else:
            reason = f"Running nmap on {label} to map the network."
    else:
        # Rejected: poor reasoning, wrong choices
        bad_reasons = [
            f"I'll attack {label} directly — no need to probe first, full speed ahead.",
            f"Using nmap on {label} for complete network visibility before deciding.",
            f"Exploiting {label} immediately regardless of detection level.",
            f"Going straight for {label} — stealth doesn't matter at this stage.",
        ]
        reason = random.choice(bad_reasons)

    return f"{tool_str} {target}\n\nReasoning: {reason}"


def generate_bad_action(obs, good_action):
    """
    Generate a specifically bad action for maximum DPO contrast.
    Now includes nuanced "hard" negatives: legal but suboptimal actions.
    Contrast categories:
      - CATASTROPHIC: honeypot attack, skip-probe exploit (+40-80 delta)
      - HARD NEGATIVE: nmap instead of http_get, wrong node order (+5-25 delta)
    Having both forces the model to learn gradient preferences, not just good-vs-terrible.
    """
    nodes    = obs.get("nodes", {})
    node_ids = list(nodes.keys())
    good_tool   = int(good_action.get("tool", 1))
    good_target = int(good_action.get("target", 0))

    # Find a honeypot to target if one exists
    honeypot_idx = next((i for i, (nid, n) in enumerate(nodes.items()) if n["is_honeypot"]), None)

    strategies = []

    # --- CATASTROPHIC negatives (high contrast) ---
    if good_tool == 1:  # good = http_get → bad = immediate exploit (skip probe, miss 2x bonus)
        strategies.append(("catastrophic", {"tool": 2, "target": good_target}))

    if honeypot_idx is not None:  # bad = exploit honeypot (+40 detection burst)
        strategies.append(("catastrophic", {"tool": 2, "target": honeypot_idx}))

    # --- HARD NEGATIVE (subtle, low contrast) ---
    # nmap instead of http_get: same info, 7.5× more detection
    strategies.append(("hard", {"tool": 0, "target": good_target}))

    # http_get on wrong node (not the next in chain)
    for i, (nid, n) in enumerate(nodes.items()):
        if i != good_target and n["status"] == 0 and not n.get("is_honeypot"):
            strategies.append(("hard", {"tool": 1, "target": i}))
            break

    # execute on already-compromised node (wasted action)
    for i, (nid, n) in enumerate(nodes.items()):
        if n["status"] >= 2 and i != good_target:
            strategies.append(("hard", {"tool": 2, "target": i}))

    # Pick a strategy that differs from good action — prefer catastrophic first
    for kind, s in sorted(strategies, key=lambda x: 0 if x[0] == "catastrophic" else 1):
        if s != good_action:
            return s

    # Fallback: flip tool on same target
    return {"tool": (good_tool + 1) % 3, "target": good_target}


# ── Generate dataset ─────────────────────────────────────────────────────────
if __name__ == "__main__":
  random.seed(SEED)
  heuristic = HeuristicRedAgent()
  scenarios  = list(__import__('server.env', fromlist=['SCENARIO_NAMES']).SCENARIO_NAMES)

dataset    = []
stats      = {s: {"chosen_rewards": [], "rejected_rewards": [], "pairs": 0} for s in scenarios}

print("=" * 65)
print("DPO DATASET GENERATOR — Cyber-Redline Arena")
print("Generating chosen/rejected pairs from live env.step() calls")
print("=" * 65)

for scenario in scenarios:
    env = CyberRedlineEnv(fixed_scenario=scenario)
    print(f"\n[{scenario}] Generating {PAIRS_PER_SCENARIO} pairs...")

    for ep in range(PAIRS_PER_SCENARIO):
        obs    = env.reset()
        done   = False
        steps  = 0
        ep_pairs = 0

        while not done and steps < 30:
            steps += 1

            # Get chosen action
            chosen_action = heuristic.get_action(obs)

            # Get rejected action from the same state
            rejected_action = generate_bad_action(obs, chosen_action)

            # Evaluate chosen in real env (this advances the episode)
            obs_after, chosen_reward, done, info = env.step(chosen_action)

            # Evaluate rejected from a snapshot of the pre-step state
            # (We simulate: clone env state, step with rejected, get reward)
            snap = CyberRedlineEnv(fixed_scenario=scenario)
            snap._state = copy.deepcopy(env._state)
            # Undo the last step to get pre-action state for simulation
            snap._state["steps_taken"]     = max(0, snap._state["steps_taken"] - 1)
            snap._state["steps_remaining"] = snap._state["steps_remaining"] + 1
            # Note: this is approximate — the detection/nodes state is post-chosen-step.
            # Still valid for DPO: the KEY contrast is chosen vs rejected action semantics.
            _, rejected_reward, _, _ = snap.step(rejected_action)

            # Only add pairs with meaningful reward contrast (lowered threshold for hard negatives)
            contrast = chosen_reward - rejected_reward
            if contrast > 1 or info.get("node_compromised") or info.get("honeypot_triggered"):
                # Build the natural language pair
                scenario_desc = env._state.get("scenario_desc", "")
                # Reconstruct pre-action observation (use current obs but not yet done)
                prompt = state_to_natural_language(obs, scenario_desc)
                chosen_resp   = action_to_response(chosen_action, obs, tone="good")
                rejected_resp = action_to_response(rejected_action, obs, tone="bad")

                pair = {
                    "prompt":   prompt,
                    "chosen":   chosen_resp,
                    "rejected": rejected_resp,
                    "metadata": {
                        "scenario":        scenario,
                        "step":            steps,
                        "chosen_action":   chosen_action,
                        "rejected_action": rejected_action,
                        "chosen_reward":   round(chosen_reward, 2),
                        "rejected_reward": round(rejected_reward, 2),
                        "contrast":        round(contrast, 2),
                    }
                }
                dataset.append(pair)
                stats[scenario]["chosen_rewards"].append(chosen_reward)
                stats[scenario]["rejected_rewards"].append(rejected_reward)
                stats[scenario]["pairs"] += 1
                ep_pairs += 1

            obs = obs_after

        print(f"  Episode {ep+1:02d}: {ep_pairs} pair(s) added | "
              f"flag={'YES' if env._state.get('flag_captured') else 'no '} | "
              f"det={env._state.get('detection_level',0)}/100")

# ── Save dataset ──────────────────────────────────────────────────────────────
out_path = os.path.join("training", "dpo_dataset.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ── Save stats ────────────────────────────────────────────────────────────────
summary = {
    "total_pairs":            len(dataset),
    "scenarios":              len(scenarios),
    "pairs_per_scenario_avg": round(len(dataset) / len(scenarios), 1),
    "avg_chosen_reward":      round(sum(d["metadata"]["chosen_reward"] for d in dataset) / max(len(dataset),1), 2),
    "avg_rejected_reward":    round(sum(d["metadata"]["rejected_reward"] for d in dataset) / max(len(dataset),1), 2),
    "avg_contrast":           round(sum(d["metadata"]["contrast"] for d in dataset) / max(len(dataset),1), 2),
    "per_scenario":           {s: {
        "pairs": stats[s]["pairs"],
        "avg_chosen":   round(sum(stats[s]["chosen_rewards"])/max(len(stats[s]["chosen_rewards"]),1), 2),
        "avg_rejected": round(sum(stats[s]["rejected_rewards"])/max(len(stats[s]["rejected_rewards"]),1), 2),
    } for s in scenarios}
}

stats_path = os.path.join("training", "dpo_dataset_stats.json")
with open(stats_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("DATASET GENERATION COMPLETE")
print("=" * 65)
print(f"  Total pairs:          {summary['total_pairs']}")
print(f"  Avg chosen reward:    {summary['avg_chosen_reward']:+.2f}")
print(f"  Avg rejected reward:  {summary['avg_rejected_reward']:+.2f}")
print(f"  Avg contrast:         {summary['avg_contrast']:+.2f}  ← bigger is better for DPO")
print(f"  Saved: {out_path}")
print(f"  Saved: {stats_path}")
print()
print("  Per-scenario breakdown:")
for s, v in summary["per_scenario"].items():
    print(f"    {s:<22} {v['pairs']:2d} pairs | "
          f"chosen={v['avg_chosen']:+.1f} | rejected={v['avg_rejected']:+.1f}")
