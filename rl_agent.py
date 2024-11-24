# rl_agent.py
import gymnasium as gym
from supply_chain_env import SupplyChainEnv
from stable_baselines3 import DQN
import numpy as np
import pandas as pd

def train_rl_agent(env, timesteps=10000):
    model = DQN('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_rl_agent(model, env, episodes=10):
    all_rewards = []
    all_info = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_rewards = []
        episode_info = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            episode_info.append(info)
        all_rewards.append(sum(episode_rewards))
        all_info.append(episode_info)
    return all_rewards, all_info

# rl_agent.py (continued)
def evaluate_rl_agent_with_logging(model, env, episodes=10):
    all_rewards = []
    all_info = []
    trajectories = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_rewards = []
        episode_info = []
        trajectory = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            episode_info.append(info)
            trajectory.append({
                'state': obs[0],
                'action': action,
                'reward': reward,
                'info': info
            })
        all_rewards.append(sum(episode_rewards))
        all_info.append(episode_info)
        trajectories.append(trajectory)
    return all_rewards, all_info, trajectories
