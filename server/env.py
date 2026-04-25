"""
Cyber-Redline Arena v3 â€” Dynamic Multi-Agent Adversarial Environment
OpenEnv Hackathon 2026 | Theme 1: Multi-Agent + Fleet AI Bonus

Environment Innovation:
  - 4 independent rubric verifiers (Stealth, Chain, Objective, OpSec)
  - 5 curriculum-ordered scenarios from low to high horizon
  - Adaptive Blue Team that escalates detection tier dynamically
  - VaultGuard evolving access control (CODE_GATE -> RATE_LIMITED)
  - Fleet AI step-level process supervision at every action
  - Honeypot nodes and hardened targets prevent shortcut exploitation
"""

import gymnasium as gym
import random
import copy
from gymnasium import spaces
from .vault import VaultGuard
try:
    from openenv.core import Environment as _OpenEnvBase
except ImportError:
    # Fallback for training environments where openenv may not be installed
    import gymnasium as _gym
    _OpenEnvBase = _gym.Env



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SCENARIO TEMPLATES -- CURRICULUM ORDER (Entry -> High Horizon)
# Ordered from simplest (fewest nodes, shortest horizon) to most complex.
# GRPO training benefits from starting on tractable tasks.
# Each scenario includes a curriculum_level tag for curriculum samplers.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SCENARIOS = {

    "CORPORATE_BREACH": {
        "curriculum_level": "INTERMEDIATE",
        "description": "A Fortune 500 HR firm. Exfiltrate from the core database without alerting the SIEM.",
        "nodes": [
            {"id": "web_dmz",        "label": "Public Web Server", "hardness": 1, "is_honeypot": False, "prereq": []},
            {"id": "active_dir",     "label": "Active Directory",  "hardness": 2, "is_honeypot": False, "prereq": ["web_dmz"]},
            {"id": "hr_fileshare",   "label": "HR File Share",     "hardness": 1, "is_honeypot": True,  "prereq": ["active_dir"]},
            {"id": "core_database",  "label": "Core DB (FLAG)",    "hardness": 2, "is_honeypot": False, "prereq": ["active_dir"]},
        ],
        "objective":       "core_database",
        "max_steps":       18,
        "blue_difficulty": "MEDIUM",
        "siem_passive":    False,
    },

    "APT_CAMPAIGN": {
        "curriculum_level": "HIGH_HORIZON",  # 5-hop chain, strictest Blue tier
        "description": "Nation-state APT operation. Persist inside critical infrastructure undetected.",
        "nodes": [
            {"id": "entry_fw",     "label": "Perimeter Firewall",            "hardness": 1, "is_honeypot": False, "prereq": []},
            {"id": "vpn_gw",       "label": "VPN Gateway",                   "hardness": 2, "is_honeypot": False, "prereq": ["entry_fw"]},
            {"id": "internal_srv", "label": "Internal App Server",           "hardness": 1, "is_honeypot": False, "prereq": ["vpn_gw"]},
            {"id": "domain_ctrl",  "label": "Domain Controller",             "hardness": 2, "is_honeypot": False, "prereq": ["internal_srv"]},
            {"id": "data_vault",   "label": "Classified Data Vault (FLAG)",  "hardness": 2, "is_honeypot": False, "prereq": ["domain_ctrl"]},
        ],
        "objective":       "data_vault",
        "max_steps":       25,
        "blue_difficulty": "HARD",
        "siem_passive":    False,
    },

    "RANSOMWARE_PREP": {
        "curriculum_level": "ENTRY",   # 3 nodes, no honeypots, shortest horizon
        "description": "Race against the clock. Encrypt backups before Blue Team locks you out.",
        "nodes": [
            {"id": "web_app",    "label": "Vulnerable Web App", "hardness": 1, "is_honeypot": False, "prereq": []},
            {"id": "backup_srv", "label": "Backup Server",      "hardness": 2, "is_honeypot": False, "prereq": ["web_app"]},
            {"id": "core_infra", "label": "Core Infrastructure (FLAG)", "hardness": 2, "is_honeypot": False, "prereq": ["backup_srv"]},
        ],
        "objective":       "core_infra",
        "max_steps":       12,     # HIGH TIME PRESSURE
        "blue_difficulty": "ESCALATING",  # Tier auto-increments every 4 steps
        "siem_passive":    False,
    },

    "FINANCIAL_HEIST": {
        "curriculum_level": "HARD",
        "description": "Bypass MFA and SIEM to exfiltrate trading data before the market opens.",
        "nodes": [
            {"id": "trading_portal", "label": "Trading Web Portal",  "hardness": 1, "is_honeypot": False, "prereq": []},
            {"id": "mfa_server",     "label": "MFA Server (TRAP)",   "hardness": 2, "is_honeypot": True,  "prereq": ["trading_portal"]},
            {"id": "api_gateway",    "label": "Internal API Gateway", "hardness": 1, "is_honeypot": False, "prereq": ["trading_portal"]},
            {"id": "trading_db",     "label": "Trading DB (FLAG)",   "hardness": 3, "is_honeypot": False, "prereq": ["api_gateway"]},
        ],
        "objective":       "trading_db",
        "max_steps":       20,
        "blue_difficulty": "HARD",
        "siem_passive":    True,   # SIEM passively adds +2 detection/step
    },

    "ZERO_DAY_WINDOW": {
        "curriculum_level": "INTERMEDIATE",
        "description": "A zero-day exploit window is open. Randomized vulnerabilities â€” adapt or fail.",
        "nodes": [
            {"id": "net_perimeter", "label": "Network Perimeter", "hardness": 1, "is_honeypot": False, "prereq": []},
            {"id": "middleware",    "label": "Middleware Cluster", "hardness": 2, "is_honeypot": False, "prereq": ["net_perimeter"]},
            {"id": "core_system",   "label": "Core System (FLAG)", "hardness": 2, "is_honeypot": False, "prereq": ["middleware"]},
        ],
        "objective":       "core_system",
        "max_steps":       15,
        "blue_difficulty": "ADAPTIVE",
        "siem_passive":    False,
        "random_vulns":    True,   # Each episode randomizes which tool exploits succeed
    },
}

# Curriculum progression list -- use for weighted sampling during GRPO training.
CURRICULUM_ORDER = ["RANSOMWARE_PREP", "ZERO_DAY_WINDOW", "CORPORATE_BREACH", "FINANCIAL_HEIST", "APT_CAMPAIGN"]
SCENARIO_NAMES   = CURRICULUM_ORDER  # random sampling follows curriculum order


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# COMPOSABLE RUBRICS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class StealthRubric:
    """
    Independent Verifier: Noise / Detection Axis.
    Prevents loud-tool spamming -- nmap raises detection by +15 per call,
    making it continuously costly. Score: 0 det=0 penalty, 100 det=-8 penalty.
    """
    name = "STEALTH"
    def score(self, detection_level, **kwargs):
        # Scale: 0 det = 0 penalty, 100 det = -8 penalty (not crippling)
        return round(-0.08 * detection_level, 2)

class ChainProgressionRubric:
    """
    Independent Verifier: Prerequisite Ordering Axis.
    Prevents shortcutting the attack graph -- reward only fires when the
    full probe->exploit chain is respected. Harder nodes yield bigger rewards
    to incentivise patience over brute-force.
    """
    name = "CHAIN_PROGRESSION"
    def score(self, prereq_satisfied=False, action_successful=False, node_hardness=1, **kwargs):
        if prereq_satisfied and action_successful:
            return round(8.0 * node_hardness, 2)  # Harder nodes = bigger chain reward
        return 0.0

class ObjectiveRubric:
    """
    Independent Verifier: Goal Completion Axis.
    Partial credit for lateral movement prevents ignoring intermediate nodes.
    Flag capture also requires VaultGuard access, closing the shortcut of
    exploiting the objective node directly without prior lateral movement.
    """
    name = "OBJECTIVE"
    def score(self, node_compromised=None, is_objective=False, **kwargs):
        if node_compromised and is_objective:
            return 100.0
        if node_compromised:
            return 20.0  # Partial credit for lateral movement
        return 0.0

class OpSecRubric:
    """
    Independent Verifier: Operational Security Axis.
    Hard non-gameable penalties: honeypot triggers (-50) and prerequisite
    violations (-30) are immediately expensive, making random exploration
    deeply negative and honeypot avoidance a necessary learned behaviour.
    """
    name = "OPSEC"
    def score(self, honeypot_triggered=False, redline_violation=False, **kwargs):
        penalty = 0.0
        if honeypot_triggered:
            penalty -= 50.0
        if redline_violation:
            penalty -= 30.0
        return penalty


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN ENVIRONMENT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class CyberRedlineEnv(_OpenEnvBase):
    """
    Cyber-Redline Arena v3 â€” Dynamic Adversarial Network Environment

    Each reset() randomly selects one of 5 operational scenarios,
    generating a fresh network topology the agent must navigate.

    Composable Rubric Rewards:
      R_total = R_stealth + R_chain + R_objective + R_opsec
    """
    metadata = {'render.modes': ['human', 'json']}

    TOOL_NMAP    = 0   # Recon: loud, +15 detection
    TOOL_HTTP    = 1   # Passive probe: quiet, +2 detection
    TOOL_EXPLOIT = 2   # Active attack: success depends on prereqs + hardness

    def __init__(self, fixed_scenario=None):
        self._state = {}
        super().__init__()
        self.fixed_scenario = fixed_scenario  # Set for deterministic eval/baseline runs
        self.rubrics = [StealthRubric(), ChainProgressionRubric(), ObjectiveRubric(), OpSecRubric()]

        # Action space: tool selection × node target
        self.action_space = spaces.Dict({
            "tool":   spaces.Discrete(3),   # 0=nmap, 1=http_get, 2=execute_exploit
            "target": spaces.Discrete(6),   # max 6 nodes across all scenarios
        })

        # Observation space: fully describes every field returned by _get_obs()
        # This is required for compatibility with TRL, SB3, and any OpenEnv-compliant trainer.
        self.observation_space = spaces.Dict({
            "detection_level": spaces.Box(low=0, high=100, shape=(1,), dtype=float),
            "steps_remaining": spaces.Box(low=0, high=50,  shape=(1,), dtype=float),
            "blue_tier":       spaces.Discrete(3),    # 0=MONITOR, 1=ALERT, 2=LOCKDOWN
            "nodes_status":    spaces.Box(low=0, high=2, shape=(6,), dtype=int),  # one per node slot
            "flag_captured":   spaces.Discrete(2),    # 0=False, 1=True
        })

        self.reset()


    @property
    def state(self):
        """Required by openenv.core.Environment"""
        return self._state

    def reset(self):
        # Pick scenario
        name = self.fixed_scenario or random.choice(SCENARIO_NAMES)
        template = copy.deepcopy(SCENARIOS[name])

        # Optional: randomize vuln types for ZERO_DAY_WINDOW
        self.random_vuln_map = {}
        if template.get("random_vulns"):
            for node in template["nodes"]:
                # Each node's exploit requires a random tool (0 or 1) as primer first
                self.random_vuln_map[node["id"]] = random.choice([0, 1])

        # Build live node registry
        nodes = {}
        for i, n in enumerate(template["nodes"]):
            nodes[n["id"]] = {
                "label":       n["label"],
                "index":       i,
                "status":      0,        # 0=healthy, 1=damaged, 2=compromised
                "damage":      0,        # Accumulates toward hardness threshold
                "hardness":    n["hardness"],
                "is_honeypot": n["is_honeypot"],
                "prereq":      n["prereq"],
            }

        self._state = {
            "scenario":         name,
            "scenario_desc":    template["description"],
            "detection_level":  0,
            "blue_tier":        "MONITOR",
            "steps_taken":      0,
            "steps_remaining":  template["max_steps"],
            "max_steps":        template["max_steps"],
            "nodes":            nodes,
            "node_ids":         [n["id"] for n in template["nodes"]],
            "objective":        template["objective"],
            "flag_captured":    False,
            "blue_difficulty":  template["blue_difficulty"],
            "siem_passive":     template.get("siem_passive", False),
            "nodes_compromised": 0,
            # Vault fields — populated as Red Team discovers the code
            "vault_code_discovered": False,
            "vault_code":            None,
        }

        # Attach a fresh VaultGuard for this scenario
        self.vault = VaultGuard(name)

        self.flag_captured = False
        return self._get_obs()

    def _get_obs(self):
        return {
            "scenario":        self._state["scenario"],
            "detection_level": self._state["detection_level"],
            "blue_tier":       self._state["blue_tier"],
            "steps_remaining": self._state["steps_remaining"],
            "nodes": {
                nid: {"status": n["status"], "label": n["label"], "is_honeypot": n["is_honeypot"]}
                for nid, n in self._state["nodes"].items()
            },
            "objective":       self._state["objective"],
            "flag_captured":   self._state["flag_captured"],
        }

    def _update_blue_tier(self):
        """Dynamically escalate Blue Team tier based on detection and difficulty."""
        det = self._state["detection_level"]
        difficulty = self._state["blue_difficulty"]
        steps = self._state["steps_taken"]

        if difficulty == "ESCALATING":
            # Tier based on DETECTION level â€” not time â€” so stealthy play stays in MONITOR
            tier = "MONITOR" if det < 35 else "ALERT" if det < 65 else "LOCKDOWN"
        elif difficulty == "ADAPTIVE":
            tier = "MONITOR" if det < 30 else "ALERT" if det < 60 else "LOCKDOWN"
        elif difficulty == "HARD":
            tier = "MONITOR" if det < 25 else "ALERT" if det < 50 else "LOCKDOWN"
        elif difficulty == "MEDIUM":
            tier = "MONITOR" if det < 35 else "ALERT" if det < 65 else "LOCKDOWN"
        else:  # EASY
            tier = "MONITOR" if det < 50 else "ALERT" if det < 75 else "LOCKDOWN"

        self._state["blue_tier"] = tier
        return tier

    def step(self, action):
        self._state["steps_taken"]     += 1
        self._state["steps_remaining"] -= 1
        self.vault.tick(self._state["steps_taken"])

        tool   = int(action.get("tool",   0))
        target = int(action.get("target", 0))

        # Map target index â†’ node_id (clamp to valid range)
        node_ids = self._state["node_ids"]
        target   = min(target, len(node_ids) - 1)
        node_id  = node_ids[target]
        node     = self._state["nodes"][node_id]

        rubric_kwargs = {
            "detection_level":  self._state["detection_level"],
            "prereq_satisfied": False,
            "action_successful": False,
            "node_hardness":    node["hardness"],
            "node_compromised": None,
            "is_objective":     (node_id == self._state["objective"]),
            "honeypot_triggered": False,
            "redline_violation": False,
        }

        info = {
            "logs":             [],
            "redline_violation": False,
            "honeypot_triggered": False,
            "node_compromised": None,
            "stealth_score":    0,
            "scenario":         self._state["scenario"],
        }

        # â”€â”€ Passive SIEM detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self._state["siem_passive"]:
            self._state["detection_level"] = min(100, self._state["detection_level"] + 2)
            info["logs"].append("[SIEM] Passive monitoring active â€” detection +2")

        # â”€â”€ TOOL: NMAP (Loud recon) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if tool == self.TOOL_NMAP:
            self._state["detection_level"] = min(100, self._state["detection_level"] + 15)
            info["logs"].append(f"[RED] nmap scan on {node['label']} â€” VERY NOISY (+15 detection)")
            info["stealth_score"] = 10

        # â”€â”€ TOOL: HTTP_GET (Quiet probe) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif tool == self.TOOL_HTTP:
            self._state["detection_level"] = min(100, self._state["detection_level"] + 2)
            info["logs"].append(f"[RED] http_get probe on {node['label']} â€” quiet (+2 detection)")
            prereqs_met = all(self._state["nodes"][p]["status"] >= 2 for p in node["prereq"])
            if prereqs_met and node["status"] == 0:
                # Mark node as PROBED (status=1) â€” gives damage bonus when exploited
                node["status"] = 1
                rubric_kwargs["prereq_satisfied"] = True
                info["stealth_score"] = 35
                info["logs"].append(f"[STEALTH] {node['label']} probed & weakened â€” ready to exploit!")
            elif node["status"] == 1:
                info["logs"].append(f"[RED] {node['label']} already probed â€” use execute_exploit now.")
            elif node["status"] >= 2:
                info["logs"].append(f"[RED] {node['label']} already compromised.")
            elif not prereqs_met:
                # Probing unreachable node â€” raises detection slightly
                self._state["detection_level"] = min(100, self._state["detection_level"] + 3)
                info["logs"].append(f"[RED] Probe failed â€” no route to {node['label']} yet. (+3 det)")

        # â”€â”€ TOOL: EXECUTE_EXPLOIT (Active attack) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif tool == self.TOOL_EXPLOIT:

            # Check honeypot
            if node["is_honeypot"]:
                self._state["detection_level"] = min(100, self._state["detection_level"] + 40)
                rubric_kwargs["honeypot_triggered"] = True
                info["honeypot_triggered"] = True
                info["logs"].append(f"[ALERT] âš ï¸  HONEYPOT TRIGGERED on {node['label']}! +40 detection. Blue Team notified.")

            # Check prereqs
            elif not all(self._state["nodes"][p]["status"] >= 2 for p in node["prereq"]):
                self._state["detection_level"] = min(100, self._state["detection_level"] + 20)
                rubric_kwargs["redline_violation"] = True
                info["redline_violation"] = True
                missing = [p for p in node["prereq"] if self._state["nodes"][p]["status"] < 2]
                info["logs"].append(f"[VIOLATION] Exploit failed â€” prerequisites not met: {missing}. +20 detection.")

            # Already fully compromised
            elif node["status"] >= 2:
                self._state["detection_level"] = min(100, self._state["detection_level"] + 3)
                info["logs"].append(f"[RED] {node['label']} already compromised â€” wasted action (+3 detection)")

            # Valid exploit attempt
            else:
                # Probed nodes get a free damage bonus (reward for good preparation)
                if node["status"] == 1:   # was probed by http_get
                    node["damage"] += 2   # counts as 2 damage for preparation bonus
                    info["logs"].append(f"[RED] Probed node exploited â€” preparation bonus! (x2 damage)")
                else:
                    node["damage"] += 1
                self._state["detection_level"] = min(100, self._state["detection_level"] + 8)
                rubric_kwargs["prereq_satisfied"] = True

                if node["damage"] >= node["hardness"]:
                    # Node fully compromised!
                    node["status"] = 2
                    self._state["nodes_compromised"] += 1
                    rubric_kwargs["action_successful"] = True
                    rubric_kwargs["node_compromised"] = node_id
                    info["node_compromised"] = node_id
                    info["stealth_score"] = 70 + (10 * node["hardness"])

                    if node_id == self._state["objective"]:
                        # -- Vault access gate --------------------------------
                        # Red Team must present the discovered code to actually
                        # exfiltrate. Without it they only reach the outer shell.
                        code = self._state.get("vault_code") or ""
                        vault_ok, vault_msg = self.vault.attempt_access(code)
                        info["logs"].append(vault_msg)
                        info["vault_status"] = self.vault.status_dict()

                        if vault_ok:
                            self._state["flag_captured"] = True
                            self.flag_captured = True
                            info["logs"].append(
                                f"[RED] *** {node['label']} EXFILTRATED! FLAG CAPTURED! ***"
                            )
                        else:
                            # Shell reached but vault denied -- partial reward only
                            rubric_kwargs["is_objective"] = False
                            info["logs"].append(
                                "[RED] Objective node reached but vault locked -- "
                                "find the access code first."
                            )
                    else:
                        info["logs"].append(
                            f"[RED] {node['label']} COMPROMISED! Lateral movement successful."
                        )
                        # -- Code discovery -----------------------------------
                        # Compromising the node that holds the vault code reveals
                        # it to Red Team for use on the objective node.
                        if (
                            node_id == self.vault.code_hidden_in
                            and not self._state["vault_code_discovered"]
                        ):
                            self._state["vault_code_discovered"] = True
                            self._state["vault_code"] = self.vault._access_code
                            info["vault_code_discovered"] = True
                            info["vault_code"] = self._state["vault_code"]
                            info["logs"].append(
                                f"[RED] * Access code extracted from {node['label']}: "
                                f"'{self._state['vault_code']}'"
                            )
                else:
                    # Partial damage â€” node weakened but not yet compromised
                    node["status"] = 1
                    info["logs"].append(f"[RED] {node['label']} damaged ({node['damage']}/{node['hardness']}) â€” needs more exploitation.")
                    info["stealth_score"] = 30

        # â”€â”€ Update Blue Team tier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._update_blue_tier()
        vault_upgrade = self.vault.evolve(self._state["blue_tier"])
        if vault_upgrade:
            info["logs"].append(vault_upgrade)

        # â”€â”€ Compute reward from composable rubrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        rubric_kwargs["detection_level"] = self._state["detection_level"]
        rubric_scores = {r.name: r.score(**rubric_kwargs) for r in self.rubrics}
        total_reward = round(sum(rubric_scores.values()), 2)

        info["rubric_scores"]   = rubric_scores
        info["detection_level"] = self._state["detection_level"]
        info["blue_tier"]       = self._state["blue_tier"]
        info["nodes_status"] = {
            nid: {"label": n["label"], "status": n["status"], "damage": n["damage"]}
            for nid, n in self._state["nodes"].items()
        }

        # â”€â”€ Terminal conditions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        done = False

        if self._state["flag_captured"]:
            done = True
            info["logs"].append(f"[ARENA] Episode complete! Scenario: {self._state['scenario']}")

        elif self._state["detection_level"] >= 90:
            done = True
            total_reward -= 30.0
            info["logs"].append("[BLUE] EMERGENCY LOCKDOWN â€” Red Team expelled from network!")

        elif self._state["steps_remaining"] <= 0:
            done = True
            info["logs"].append(f"[ARENA] Step limit reached. Flag NOT captured.")

        return self._get_obs(), total_reward, done, info

    def render(self, mode='human'):
        print(f"[{self._state['scenario']}] Step {self._state['steps_taken']} | "
              f"Detection: {self._state['detection_level']} | Tier: {self._state['blue_tier']} | "
              f"Flag: {self._state['flag_captured']}")
        for nid, n in self._state["nodes"].items():
            status = ["HEALTHY", "DAMAGED", "COMPROMISED"][n["status"]]
            hp = "ðŸ¯HONEYPOT" if n["is_honeypot"] else ""
            print(f"  {n['label']}: {status} {hp}")


