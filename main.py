# main.py

import warnings

# Suppress the specific warning from torchvision
warnings.filterwarnings(
    "ignore",
    message=".*Failed to load image Python extension.*"
)

from supply_chain_env import SupplyChainEnv
from cbc_solution import cbc_optimize
from rl_agent import train_rl_agent, evaluate_rl_agent_with_logging
from decision_transformer import train_decision_transformer, evaluate_decision_transformer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    env = SupplyChainEnv()
    
    # CBC Solution
    optimal_orders, demand = cbc_optimize(env)
    cbc_rewards = []
    env.reset()
    for t in range(env.max_steps):
        action = optimal_orders[t]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cbc_rewards.append(reward)
        if done:
            break
    
    cbc_total_profit = sum(cbc_rewards)
    
    # RL Agent
    rl_model = train_rl_agent(env)
    rl_rewards, rl_info, rl_trajectories = evaluate_rl_agent_with_logging(rl_model, env)
    rl_total_profit = sum(rl_rewards) / len(rl_rewards)
    # print("RL Trajectories:")
    # for i, traj in enumerate(rl_trajectories):
    #     print(f"Trajectory {i}: {traj}")
    #     print(f"Actions: {[step['action'] for step in traj]}")
    # Train Decision Transformer
    dt_model = train_decision_transformer(rl_trajectories)
    dt_rewards, dt_info = evaluate_decision_transformer(dt_model, env)
    dt_total_profit = sum(dt_rewards) / len(dt_rewards)
    
    # DataFrame for Visualization
    data = pd.DataFrame({
        'CBC': [cbc_total_profit],
        'RL Agent': [rl_total_profit],
        'Decision Transformer': [dt_total_profit]
    })
    
    # Visualization
    sns.barplot(data=data)
    plt.title('Total Profit Comparison')
    plt.ylabel('Total Profit')
    plt.show()
    
if __name__ == '__main__':
    main()
