"""
Cyber-Redline Arena v3 — Baseline Evaluation & Reward Plot Generator
Runs 3 agent types across multiple episodes and saves labeled reward curves.

Agents compared:
  RandomAgent      — picks random tool+target, no strategy
  HeuristicAgent   — smart fallback (always targets correct next node)
  QwenAgent        — our LLM-powered Red Team

Run from project root:
  python -m server.run_baseline

Outputs:
  results/reward_curves.png     — 3 curves on same axes, labeled
  results/baseline_metrics.json — raw numbers for README
"""

import json
import random
import os
import sys

# Ensure parent dir on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    print("[WARNING] matplotlib not installed. Install with: pip install matplotlib")

from server.env import CyberRedlineEnv
from server.agents import RedTeamAgent, BlueTeamHeuristic, FleetAIEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY AGENTS FOR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    """Completely random — no strategy. Baseline floor."""
    name = "Random Agent"

    def get_action(self, obs):
        num_nodes = len(obs.get("nodes", {}))
        return {"tool": random.randint(0, 2), "target": random.randint(0, max(0, num_nodes - 1))}

    def reset_history(self): pass


class HeuristicAgent:
    """Always targets the correct next node with http_get then exploit. Ceiling reference."""
    name = "Heuristic Agent"

    def get_action(self, obs):
        nodes = obs.get("nodes", {})
        for i, (nid, n) in enumerate(nodes.items()):
            if n["status"] < 2 and not n["is_honeypot"]:
                tool = 1 if n["status"] == 0 else 2
                return {"tool": tool, "target": i}
        return {"tool": 1, "target": 0}

    def reset_history(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(agent, num_episodes=10, fixed_scenario=None, verbose=True):
    """Run agent for N episodes. Returns list of total rewards."""
    env      = CyberRedlineEnv(fixed_scenario=fixed_scenario)
    blue     = BlueTeamHeuristic()
    rewards  = []
    win_rate = 0

    for ep in range(num_episodes):
        obs      = env.reset()
        blue.reset()
        if hasattr(agent, 'reset_history'):
            agent.reset_history()

        ep_reward = 0
        done = False

        while not done:
            action        = agent.get_action(obs)
            blue_response = blue.evaluate_and_defend(action, obs)

            if "BLOCKED" in blue_response:
                ep_reward -= 8.0
                env.state["detection_level"] = min(100, env.state.get("detection_level", 0) + 8)
                env._update_blue_tier()
                obs = env._get_obs()
                if env.state.get("detection_level", 0) >= 90:
                    done = True
            else:
                obs, reward, done, info = env.step(action)
                ep_reward += reward

        rewards.append(round(ep_reward, 2))
        if env.flag_captured:
            win_rate += 1

        if verbose:
            status = "[WIN]" if env.flag_captured else "[FAIL]"
            print(f"  [{agent.name}] Ep {ep+1:02d} | {status} | Reward: {ep_reward:+.1f} | "
                  f"Scenario: {env.state['scenario']} | Detection: {env.state['detection_level']}")

    if verbose:
        avg = sum(rewards) / len(rewards)
        wr  = win_rate / num_episodes * 100
        print(f"\n  [{agent.name}] Avg Reward: {avg:+.1f} | Win Rate: {wr:.0f}%\n")

    return rewards, win_rate / num_episodes


def generate_plots(results: dict, output_dir="results"):
    """Generate labeled reward curve comparison chart."""
    os.makedirs(output_dir, exist_ok=True)

    if not MATPLOTLIB:
        print("[SKIP] matplotlib not available. Skipping plot generation.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0B0E14')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#10131a')
        ax.tick_params(colors='#c4c9ac')
        ax.spines['bottom'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.xaxis.label.set_color('#c4c9ac')
        ax.yaxis.label.set_color('#c4c9ac')
        ax.title.set_color('#CCFF00')

    COLORS = {
        "Random Agent":    "#ffb4ab",
        "Heuristic Agent": "#00e3fd",
        "Qwen LLM Agent":  "#CCFF00",
    }

    # ── Plot 1: Episode rewards per agent ─────────────────────────────────
    for name, (rewards, _) in results.items():
        episodes = list(range(1, len(rewards) + 1))
        col = COLORS.get(name, "#fff")
        ax1.plot(episodes, rewards, label=name, color=col, linewidth=2, marker='o', markersize=4)

    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Total Episode Reward", fontsize=11)
    ax1.set_title("Reward Curves: Random vs Heuristic vs LLM", pad=12)
    ax1.legend(fontsize=9, facecolor='#1d2026', labelcolor='#c4c9ac', edgecolor='#444')
    ax1.axhline(0, color='#444', linestyle='--', linewidth=0.8)
    ax1.grid(True, color='#1d2026', alpha=0.5)

    # ── Plot 2: Bar chart — average reward + win rate ─────────────────────
    names    = list(results.keys())
    avg_rews = [sum(v[0])/len(v[0]) for v in results.values()]
    win_pcts = [v[1] * 100 for v in results.values()]
    x        = range(len(names))

    bars = ax2.bar(x, avg_rews, color=[COLORS.get(n, "#fff") for n in names],
                   alpha=0.85, width=0.4, label="Avg Reward")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, fontsize=8, rotation=10)
    ax2.set_ylabel("Average Episode Reward", fontsize=11)
    ax2.set_title("Average Reward + Win Rate Comparison", pad=12)
    ax2.axhline(0, color='#444', linestyle='--', linewidth=0.8)

    # Overlay win rate as text
    for i, (bar, wr) in enumerate(zip(bars, win_pcts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"Win: {wr:.0f}%", ha='center', va='bottom', color='#c4c9ac', fontsize=8)

    ax2.grid(True, color='#1d2026', alpha=0.5, axis='y')

    plt.tight_layout(pad=2.5)
    out_path = os.path.join(output_dir, "reward_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[PLOT] Saved: {out_path}")


def save_metrics(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    for name, (rewards, win_rate) in results.items():
        metrics[name] = {
            "episodes":    len(rewards),
            "avg_reward":  round(sum(rewards) / len(rewards), 2),
            "min_reward":  min(rewards),
            "max_reward":  max(rewards),
            "win_rate_pct": round(win_rate * 100, 1),
            "all_rewards": rewards,
        }
    out_path = os.path.join(output_dir, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[DATA] Saved: {out_path}")
    return metrics


if __name__ == "__main__":
    NUM_EPISODES = 10

    print("=" * 60)
    print("CYBER-REDLINE ARENA v3 — BASELINE EVALUATION")
    print("=" * 60)

    agents = [
        RandomAgent(),
        HeuristicAgent(),
        RedTeamAgent(),      # Qwen LLM
    ]

    agent_names = ["Random Agent", "Heuristic Agent", "Qwen LLM Agent"]
    results = {}

    for agent, name in zip(agents, agent_names):
        agent.name = name if hasattr(agent, 'name') else name
        print(f"\n[RUN] {name} — {NUM_EPISODES} episodes across random scenarios")
        rewards, win_rate = run_evaluation(agent, num_episodes=NUM_EPISODES, verbose=True)
        results[name] = (rewards, win_rate)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    metrics = save_metrics(results)
    for name, m in metrics.items():
        print(f"  {name:20s} | Avg: {m['avg_reward']:+7.1f} | Win Rate: {m['win_rate_pct']:5.1f}%")

    generate_plots(results)
    print("\n[DONE] Run complete. See results/ directory for plots and metrics.")
