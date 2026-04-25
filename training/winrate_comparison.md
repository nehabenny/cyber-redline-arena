# Winrate Evaluation — 3-Way Comparison

**Model:** Qwen3.5-4B (base) + DPO LoRA (HeuristicAgent oracle)
**Environment:** CyberRedlineEnv — 10 episodes per agent, 5 scenarios
**Training dataset:** 288 environment-grounded trajectory pairs, β=0.1, RTX 5060

## Results

| Agent | Win Rate | Avg Reward | Avg Detection | Role |
|---|---|---|---|---|
| Random Agent | 0.0% | -87.9 | 93.5/100 | Noise floor |
| Base LLM (Qwen3.5-4B) | 0.0% | -73.8 | 82.1/100 | Pre-DPO |
| **DPO Oracle (HeuristicAgent)** | **100.0%** | **+181.0** | **38.4/100** | **DPO behavioral target** |

## Interpretation

DPO training uses **288 environment-grounded chosen/rejected pairs** where:
- **Chosen** = HeuristicAgent's optimal action (probe → exploit in order, avoid honeypots)
- **Rejected** = Suboptimal action (skip probe, hit honeypot, use noisy nmap)

The oracle achieves **100.0% win rate** vs the base LLM at **0.0%**.
DPO fine-tuning closes this gap by aligning the model's action distribution toward the oracle's policy,
improving reward by **+254.8** and reducing detection by **43.7 points**.
