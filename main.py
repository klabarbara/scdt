# main.py
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
        obs, reward, done, info = env.step(action)
        cbc_rewards.append(reward)
    
    cbc_total_profit = sum(cbc_rewards)
    
    # RL Agent
    rl_model = train_rl_agent(env)
    rl_rewards, rl_info, rl_trajectories = evaluate_rl_agent_with_logging(rl_model, env)
    rl_total_profit = sum(rl_rewards) / len(rl_rewards)
    
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
    
def visualize_metrics(cbc_info, rl_info, dt_info):
    # Extract metrics
    cbc_inventory = [info['inventory_level'] for info in cbc_info]
    rl_inventory = [step['info']['inventory_level'] for traj in rl_info for step in traj]
    dt_inventory = [info['inventory_level'] for info in dt_info]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Step': list(range(len(cbc_inventory))),
        'CBC': cbc_inventory,
        'RL Agent': rl_inventory[:len(cbc_inventory)],
        'Decision Transformer': dt_inventory
    })
    df_melted = df.melt('Step', var_name='Method', value_name='Inventory Level')
    
    # Plot
    sns.lineplot(data=df_melted, x='Step', y='Inventory Level', hue='Method')
    plt.title('Inventory Level Over Time')
    plt.show()

if __name__ == '__main__':
    main()
