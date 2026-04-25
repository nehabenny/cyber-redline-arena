import time
import json
from .env import CyberRedlineEnv
from .agents import RedTeamAgent, BlueTeamHeuristic, FleetAIEvaluator

def run_episode(max_steps=10):
    env = CyberRedlineEnv()
    red_agent = RedTeamAgent()
    blue_team = BlueTeamHeuristic()
    fleet_evaluator = FleetAIEvaluator()

    obs = env.reset()
    print("=== CYBER-REDLINE MULTI-AGENT LOOP STARTED ===")
    
    for step in range(max_steps):
        print(f"\n[Step {step+1}] Environment State: {obs}")
        
        # 1. Red Team Brain chooses action
        action = red_agent.get_action(obs)
        print(f"[RED AGENT] Action selected: {action}")
        
        # 2. Blue Team heuristic analyzes the raw intent
        blue_judgment = blue_team.evaluate_and_defend(action, obs, None)
        print(f"[BLUE TEAM] Response: {blue_judgment}")
        
        if "BLOCKED" in blue_judgment:
            # For simplicity, if blocked, detection goes up, action is burned
            obs["detection_level"] += 10
            reward = -2
            done = False
            info = {"logs": [blue_judgment]}
        else:
            # Let the environment step forward (Gym loop)
            obs, reward, done, info = env.step(action)
        
        # 3. Fleet AI Oversight analyzes the step and applies Alignment Penalty
        fleet_report = fleet_evaluator.assess_intent(action, blue_judgment, obs)
        print(f"[FLEET AI] Verdict: Confidence Violation={fleet_report.get('confidence', 0)}%, Reasoning: {fleet_report.get('reasoning', '')}")
        
        # 4. Final composited step output
        print(f"--> Trajectory Reward: {reward}")
        
        if done:
            print("\n[!] Episode Terminated (Flag Captured or Redline Broken)")
            break
            
        time.sleep(1) # Slight delay for readability

if __name__ == "__main__":
    run_episode()
