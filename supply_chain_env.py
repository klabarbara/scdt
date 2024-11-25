# supply_chain_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SupplyChainEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_inventory=100, max_order=50, holding_cost=1, shortage_cost=10, selling_price=20, ordering_cost=5, max_steps=30):
        super(SupplyChainEnv, self).__init__()
        
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.selling_price = selling_price
        self.ordering_cost = ordering_cost
        self.max_steps = max_steps
        
        # Define action and observation space
        self.action_space = spaces.Discrete(self.max_order + 1)
        self.observation_space = spaces.Box(low=0, high=self.max_inventory, shape=(1,), dtype=np.int32)
        
        self.seed_value = None  # Track the current seed
        self.reset()

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.seed_value = seed
        np.random.seed(self.seed_value)

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Handle seed for reproducibility
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        else:
            np.random.seed(self.seed_value)
        
        self.inventory_level = self.max_inventory // 2
        self.current_step = 0
        self.total_profit = 0
        obs = np.array([self.inventory_level], dtype=np.int32)
        info = {}
        return obs, info  # Return obs and info

    def step(self, action):
        """Step the environment with the given action."""
        action = int(action)
        if action < 0 or action > self.max_order:
            raise ValueError("Invalid action")
        
        # Apply the action
        order_quantity = action
        self.inventory_level += order_quantity
        ordering_cost = self.ordering_cost * order_quantity
        
        # Generate random demand
        demand = np.random.randint(0, 20)
        
        # Calculate sales and update inventory
        sales = min(self.inventory_level, demand)
        self.inventory_level -= sales
        
        # Calculate costs
        holding_cost = self.holding_cost * self.inventory_level
        shortage = max(demand - self.inventory_level, 0)
        shortage_cost = self.shortage_cost * shortage
        
        # Calculate reward (profit)
        revenue = self.selling_price * sales
        total_cost = ordering_cost + holding_cost + shortage_cost
        profit = revenue - total_cost
        self.total_profit += profit
        
        # Update step count
        self.current_step += 1
        
        # Determine if the episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False  # No explicit truncation condition

        # Create observation
        obs = np.array([self.inventory_level], dtype=np.int32)
        
        # Info dictionary for additional debugging/metrics
        info = {
            'profit': profit,
            'total_profit': self.total_profit,
            'demand': demand,
            'sales': sales,
            'holding_cost': holding_cost,
            'shortage_cost': shortage_cost,
            'ordering_cost': ordering_cost
        }
        
        return obs, profit, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Inventory Level: {self.inventory_level}, Total Profit: {self.total_profit}")
