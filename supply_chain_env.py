# supply_chain_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
        # Actions: reorder quantity (0 to max_order)
        self.action_space = spaces.Discrete(self.max_order + 1)
        
        # Observations: current inventory level (0 to max_inventory)
        self.observation_space = spaces.Box(low=0, high=self.max_inventory, shape=(1,), dtype=np.int32)
        
        self.reset()
        
    def reset(self):
        self.inventory_level = self.max_inventory // 2
        self.current_step = 0
        self.total_profit = 0
        return np.array([self.inventory_level], dtype=np.int32)
    
    def step(self, action):
        # Ensure action is valid
        action = int(action)
        if action < 0 or action > self.max_order:
            raise ValueError("Invalid action")
        
        # Place order
        order_quantity = action
        self.inventory_level += order_quantity
        ordering_cost = self.ordering_cost * order_quantity
        
        # Random demand
        demand = np.random.randint(0, 20)
        
        # Sales and update inventory
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
        
        # Update step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Observation
        obs = np.array([self.inventory_level], dtype=np.int32)
        
        # Info dictionary for logging
        info = {
            'profit': profit,
            'total_profit': self.total_profit,
            'demand': demand,
            'sales': sales,
            'holding_cost': holding_cost,
            'shortage_cost': shortage_cost,
            'ordering_cost': ordering_cost
        }
        
        return obs, profit, done, info
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Inventory Level: {self.inventory_level}, Total Profit: {self.total_profit}")
