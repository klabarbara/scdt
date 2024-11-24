# decision_transformer.py
from transformers import DecisionTransformerModel, DecisionTransformerConfig
import torch
from torch.utils.data import Dataset, DataLoader

class SupplyChainDataset(Dataset):
    def __init__(self, trajectories, max_length):
        self.trajectories = trajectories
        self.max_length = max_length
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = [step['state'] for step in traj]
        actions = [step['action'] for step in traj]
        rewards = [step['reward'] for step in traj]
        
        # Padding
        padding_length = self.max_length - len(states)
        if padding_length > 0:
            states += [0] * padding_length
            actions += [0] * padding_length
            rewards += [0] * padding_length
        
        return {
            'states': torch.tensor(states[:self.max_length], dtype=torch.long),
            'actions': torch.tensor(actions[:self.max_length], dtype=torch.long),
            'rewards': torch.tensor(rewards[:self.max_length], dtype=torch.float)
        }

def train_decision_transformer(trajectories, max_length=30, epochs=10, batch_size=8):
    dataset = SupplyChainDataset(trajectories, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    config = DecisionTransformerConfig(
        state_dim=1,
        act_dim=1,
        hidden_size=64,
        max_length=max_length,
        n_layer=2,
        n_head=2,
        n_inner=128,
        activation_function='relu',
        n_positions=max_length,
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )
    
    model = DecisionTransformerModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            states = batch['states'].unsqueeze(-1).float()
            actions = batch['actions'].unsqueeze(-1).float()
            rewards = batch['rewards']
            
            outputs = model(states=states, actions=actions, returns_to_go=rewards)
            action_preds = outputs.logits.squeeze(-1)
            
            loss = loss_fn(action_preds, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

# decision_transformer.py (continued)
def evaluate_decision_transformer(model, env, episodes=10):
    all_rewards = []
    all_info = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_rewards = []
        episode_info = []
        states = []
        actions = []
        rewards = []
        while not done:
            state_tensor = torch.tensor([[obs[0]]], dtype=torch.float)
            if len(actions) == 0:
                action = env.action_space.sample()
            else:
                past_states = torch.tensor([states], dtype=torch.float)
                past_actions = torch.tensor([actions], dtype=torch.float)
                past_rewards = torch.tensor([rewards], dtype=torch.float)
                outputs = model(
                    states=past_states.unsqueeze(-1),
                    actions=past_actions.unsqueeze(-1),
                    returns_to_go=past_rewards
                )
                action_logits = outputs.logits.squeeze(-1)
                action = int(torch.argmax(action_logits[-1]).item())
            obs, reward, done, info = env.step(action)
            states.append(obs[0])
            actions.append(action)
            rewards.append(reward)
            episode_rewards.append(reward)
            episode_info.append(info)
        all_rewards.append(sum(episode_rewards))
        all_info.append(episode_info)
    return all_rewards, all_info
