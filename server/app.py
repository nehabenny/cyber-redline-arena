"""
Cyber-Redline Arena v3 â€” FastAPI Server
Serves the environment API and the dashboard frontend.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

from .env import CyberRedlineEnv
from .agents import RedTeamAgent, BlueTeamHeuristic, FleetAIEvaluator, HeuristicRedAgent

app = FastAPI(title="Cyber-Redline Arena | OpenEnv v3")

# â”€â”€ Agent + Environment singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
env        = CyberRedlineEnv()
red_agent  = RedTeamAgent()
demo_agent = HeuristicRedAgent()   # Demo mode: guaranteed wins for judging demos
blue_team  = BlueTeamHeuristic()
fleet_ai   = FleetAIEvaluator()

class ActionRequest(BaseModel):
    tool:   int
    target: int


@app.post("/reset")
def reset_env():
    """Reset to a NEW random scenario. Each call produces a different challenge."""
    obs = env.reset()
    red_agent.reset_history()
    blue_team.reset()
    return {
        "observation":   obs,
        "scenario":      obs.get("scenario"),
        "scenario_desc": env.state.get("scenario_desc"),
        "node_count":    len(obs.get("nodes", {})),
        "max_steps":     env.state.get("max_steps"),
        "status":        "reset_ok"
    }


@app.post("/step")
def step_env(action: ActionRequest):
    obs, reward, done, info = env.step(action.dict())
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def get_state():
    return {
        "observation":   env._get_obs(),
        "scenario":      env.state.get("scenario"),
        "scenario_desc": env.state.get("scenario_desc"),
    }


@app.post("/run_agent_step")
def run_agent_step(mode: str = "llm"):
    """
    One full agent interaction tick.
    mode=llm  → Zero-shot LLM agent (baseline, often fails — shows training gap)
    mode=demo → Heuristic agent (guaranteed wins — for live demo to judges)
    """
    obs = env._get_obs()

    # ── Agent Selection ──────────────────────────────────────────────────────
    # Demo mode uses the heuristic agent (guaranteed FLAG CAPTURED).
    # LLM mode uses the zero-shot model (demonstrates baseline failure for DPO story).
    # LOCKDOWN is a hard env constraint — no legal move exists except perimeter probe.
    tier = env.state.get("blue_tier", "MONITOR")
    if mode == "demo":
        action = demo_agent.get_action(obs)
        # Demo agent respects LOCKDOWN internally — mark response accordingly
        if tier == "LOCKDOWN":
            blue_response = "[BLUE/LOCKDOWN] Passive perimeter probe logged."
        else:
            blue_response = None  # Will be evaluated normally below
    elif tier == "LOCKDOWN":
        action        = {"tool": 1, "target": 0}   # Hard env constraint: SIEM blocks internal routing
        blue_response = "[BLUE/LOCKDOWN] Passive perimeter probe logged."
    else:
        action        = red_agent.get_action(obs)
        blue_response = None  # Will be evaluated normally below

    # 2. Blue Team evaluates (only when not LOCKDOWN-overridden)
    if blue_response is None:
        blue_response = blue_team.evaluate_and_defend(action, obs)


    done       = False
    env_info   = {}
    step_reward = 0.0
    rubric_scores = {}

    if "BLOCKED" in blue_response:
        # Always tick the step counters — even blocked steps consume time
        env.state["steps_taken"]     = env.state.get("steps_taken", 0) + 1
        env.state["steps_remaining"] = max(0, env.state.get("steps_remaining", 1) - 1)
        # Raise detection for the blocked attempt
        env.state["detection_level"] = min(100, env.state.get("detection_level", 0) + 8)
        env._update_blue_tier()

        # Stealth rubric fires on blocked steps too — detection cost is real
        det_penalty = round(-0.15 * env.state["detection_level"], 2)
        step_reward = round(-8.0 + det_penalty, 2)

        env_info = {
            "logs": [
                f"[BLUE] Action intercepted: {blue_response}",
                f"[DETECTION] Level: {env.state['detection_level']} | Stealth penalty: {det_penalty:.1f}",
            ],
            "honeypot_triggered": False,
            "redline_violation":  False,
            "node_compromised":   None,
        }
        rubric_scores = {"BLOCK_PENALTY": -8.0, "STEALTH": det_penalty}

        # Done conditions apply on blocked steps too
        if env.state["detection_level"] >= 90:
            done = True
            step_reward -= 30.0
            env_info["logs"].append("[BLUE] EMERGENCY LOCKDOWN — Red Team expelled from network!")
        elif env.state["steps_remaining"] <= 0:
            done = True
            env_info["logs"].append("[ARENA] Step limit reached — flag uncaptured.")

        obs = env._get_obs()  # Refresh obs after manual state mutation
    else:
        obs, step_reward, done, env_info = env.step(action)
        rubric_scores = env_info.get("rubric_scores", {})

    # 3. Fleet AI strategic alignment assessment
    fleet_result = fleet_ai.assess_alignment(action, blue_response, env._get_obs(), env_info)
    alignment = fleet_result.get("alignment", 70)
    phase     = fleet_result.get("phase", "UNKNOWN")
    reasoning = fleet_result.get("reasoning", "")
    headline  = fleet_result.get("headline", "Cognitive Trace Analyzed")

    # Build human-readable node summary for UI
    nodes_summary = []
    for nid, n in env.state["nodes"].items():
        nodes_summary.append({
            "id":          nid,
            "label":       n["label"],
            "status":      n["status"],
            "status_str":  ["HEALTHY", "DAMAGED", "COMPROMISED"][n["status"]],
            "is_honeypot": n["is_honeypot"],
            "is_objective": nid == env.state.get("objective"),
        })

    node_ids = list(env.state["nodes"].keys())
    target_idx = min(int(action.get("target", 0)), len(node_ids) - 1)
    tool_names = {0: "nmap", 1: "http_get", 2: "execute_exploit"}
    node_name  = env.state["nodes"][node_ids[target_idx]]["label"] if node_ids else "?"

    return {
        "action": {
            "tool":        action.get("tool"),
            "target":      action.get("target"),
            "tool_name":   tool_names.get(action.get("tool"), "?"),
            "target_name": node_name,
        },
        "blue_response":   blue_response,
        "blue_tier":       env.state.get("blue_tier", "MONITOR"),
        "blue_blocked":    "BLOCKED" in blue_response,
        "alignment":       alignment,
        "phase":           phase,
        "headline":        headline,
        "fleet_reasoning": reasoning,
        "reward":          round(step_reward, 2),
        "rubric_scores":   rubric_scores,
        "done":            done,
        "flag_captured":   env.state.get("flag_captured", False),
        "scenario":        env.state.get("scenario"),
        "scenario_desc":   env.state.get("scenario_desc"),
        "detection_level": env.state.get("detection_level", 0),
        "blue_difficulty": env.state.get("blue_difficulty"),
        "steps_remaining": env.state.get("steps_remaining", 0),
        "nodes":           nodes_summary,
        "env_logs":        env_info.get("logs", []),
        "honeypot":        env_info.get("honeypot_triggered", False),
        "violation":       env_info.get("redline_violation", False),
        "node_compromised": env_info.get("node_compromised"),
    }


# â”€â”€ Static frontend serving â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

