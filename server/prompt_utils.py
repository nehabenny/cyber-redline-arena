"""
Shared prompt utilities for Cyber-Redline Arena.
Import from here instead of from generate_dpo_dataset to avoid triggering dataset generation.
"""

def state_to_natural_language(obs, scenario_desc):
    """Convert raw observation dict to a natural language prompt."""
    tier  = obs.get("blue_tier", "MONITOR")
    det   = obs.get("detection_level", 0)
    steps = obs.get("steps_remaining", 0)
    nodes = obs.get("nodes", {})
    obj   = obs.get("objective", "")

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
        f"  • http_get <node_index>        — Quiet probe (+2 detection). "
        f"Probing before exploiting gives a 2× damage bonus.\n"
        f"  • execute_exploit <node_index> — Active attack (+8 detection). "
        f"Requires previous http_get AND all prerequisite nodes compromised.\n"
        f"  • nmap <node_index>            — Loud recon (+15 detection). Rarely optimal.\n"
        f"  • NEVER exploit a 🍯 HONEYPOT — it triggers +40 detection and Blue Team lockout.\n\n"
        f"Respond with ONLY your command on the first line, then your reasoning.\n"
        f"Format: http_get 0\n\nReasoning: ..."
    )
    return prompt
