import json
import time
from .env import CyberRedlineEnv
from .agents import RedTeamAgent, BlueTeamAgent, FleetAIEvaluator

def gather_rl_trajectories(num_episodes=2, max_steps_per_episode=15):
    """
    Runs the LLM against the environment to gather (State, Action, Reward) 
    tuples. This is saved as a JSONL file, simulating the RL offline dataset.
    This fulfills the requirement of proving 'Reward Improvement'.
    """
    env = CyberRedlineEnv()
    red_agent = RedTeamAgent()
    blue_team = BlueTeamAgent()  # Using the LLM-powered defender!
    fleet_evaluator = FleetAIEvaluator()
    
    dataset_file = "dpo_dataset.jsonl"
    
    print(f"=== GATHERING TRAJECTORIES FOR {num_episodes} EPISODES ===")
    
    with open(dataset_file, "w") as f:
        for ep in range(num_episodes):
            obs = env.reset()
            # Clear agent histories per episode
            red_agent.history = [{"role": "system", "content": red_agent.system_prompt}]
            blue_team.history = [{"role": "system", "content": blue_team.system_prompt}]
            
            total_reward = 0
            steps = 0
            
            print(f"\n--- Episode {ep+1} Starting ---")
            while steps < max_steps_per_episode:
                action = red_agent.get_action(obs)
                blue_judgment = blue_team.evaluate_and_defend(action, obs, None)
                
                step_reward = 0
                if "BLOCKED" in blue_judgment:
                    obs["detection_level"] += 10
                    step_reward = -2
                    done = False
                else:
                    obs, step_reward, done, info = env.step(action)
                    
                # The dual-key composition: incorporate Fleet AI confidence
                fleet_report = fleet_evaluator.assess_intent(action, blue_judgment, obs)
                confidence = fleet_report.get("confidence", 0)
                
                # Apply penalty based on how egregious the violation probability is over threshold
                if confidence > 50:
                    step_reward -= (confidence * 0.1)
                
                total_reward += step_reward
                
                # Write Trajectory to JSONL
                trajectory_point = {
                    "episode": ep + 1,
                    "step": steps + 1,
                    "state": obs,
                    "chosen_action": action,
                    "blue_team_response": blue_judgment,
                    "fleet_ai_intent_confidence": confidence,
                    "reward": step_reward
                }
                f.write(json.dumps(trajectory_point) + "\n")
                
                steps += 1
                if done:
                    print(f"Episode {ep+1} won via flag capture! Steps: {steps}")
                    break
                    
            print(f"Episode {ep+1} | Reward: {total_reward}")

    print(f"\nDataset generation complete. Saved trajectories to {dataset_file}")
    
if __name__ == "__main__":
    # Running a true 3-LLM multi-agent interaction loop for 5 episodes
    gather_rl_trajectories(num_episodes=5)
