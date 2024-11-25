# decision_transformer.py
from transformers import DecisionTransformerModel, DecisionTransformerConfig
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np  

class SupplyChainDataset(Dataset):
    def __init__(self, trajectories, max_length):
        self.trajectories = trajectories
        self.max_length = max_length
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        states = [step['state'] for step in trajectory]
        actions = [
            step['action'].item() if hasattr(step['action'], 'item') else int(step['action'])
            for step in trajectory
        ]
        rewards = [step['reward'] for step in trajectory]

        # Compute returns-to-go
        returns_to_go = np.cumsum(rewards[::-1])[::-1].tolist()

        seq_length = len(states)
        if seq_length < self.max_length:
            padding_length = self.max_length - seq_length
            states += [0] * padding_length
            actions += [0] * padding_length
            rewards += [0] * padding_length
            returns_to_go += [0] * padding_length
            mask = [1] * seq_length + [0] * padding_length
        else:
            states = states[:self.max_length]
            actions = actions[:self.max_length]
            rewards = rewards[:self.max_length]
            returns_to_go = returns_to_go[:self.max_length]
            mask = [1] * self.max_length

        return {
            'states': torch.tensor(states, dtype=torch.float),
            'actions': torch.tensor(actions, dtype=torch.float),
            'rewards': torch.tensor(rewards, dtype=torch.float),
            'returns_to_go': torch.tensor(returns_to_go, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float)
        }



def train_decision_transformer(trajectories, max_length=30, epochs=10, batch_size=8):
    # Instantiate the dataset
    dataset = SupplyChainDataset(trajectories, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = DecisionTransformerConfig(
        state_dim=1,
        act_dim=1,  # For continuous action values
        hidden_size=64,
        max_length=max_length,
        n_layer=2,
        n_head=2,
        n_inner=128,
        activation_function='relu',
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        output_hidden_states=False,
        use_return_embeddings=True  # Ensure return embeddings are used correctly
    )

    model = DecisionTransformerWithActionHead(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction='none')  # Set reduction to 'none' to get per-element loss

    for epoch in range(epochs):
        for batch in dataloader:
            states = batch['states'].unsqueeze(-1).float()        # (batch_size, seq_length, 1)
            actions = batch['actions'].unsqueeze(-1).float()      # (batch_size, seq_length, 1)
            returns_to_go = batch['returns_to_go'].unsqueeze(-1).float()  # (batch_size, seq_length, 1)
            mask = batch['mask']  # (batch_size, seq_length)

            batch_size, seq_length, _ = states.size()
            timesteps = torch.arange(seq_length, device=states.device).unsqueeze(0).repeat(batch_size, 1)

            # Shift actions to get input actions (actions at time t-1)
            input_actions = torch.zeros_like(actions)
            input_actions[:, 1:, :] = actions[:, :-1, :]
            input_actions[:, 0, :] = 0  # Or use a special token or initial action

            # Forward pass
            action_preds = model(
                states=states,
                actions=input_actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps
            )  # action_preds shape: (batch_size, seq_length, act_dim)

            # Targets are the actions at time t
            action_target = actions  # (batch_size, seq_length, act_dim)

            # Compute loss only over valid positions
            loss = loss_fn(action_preds.squeeze(-1), action_target.squeeze(-1))  # (batch_size, seq_length)
            loss = loss * mask  # Apply mask
            loss = loss.sum() / mask.sum()  # Average over valid positions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_decision_transformer(model, env, max_length=30, episodes=10):
    all_rewards = []
    all_info = []
    for episode in range(episodes):
        obs, _ = env.reset()
        if isinstance(obs, np.ndarray):
            obs = obs.item()
        done = False
        episode_rewards = []
        episode_info = []
        states = [obs]  # Start with initial observation
        actions = []
        rewards = []
        timesteps = [0]
        step = 0
        while not done and step < max_length:
            if len(actions) == 0:
                action_value = env.action_space.sample()
            else:
                # Convert current observation to scalar if necessary
                if isinstance(obs, np.ndarray):
                    obs = obs.item()

                # Prepare inputs for the model
                # Exclude the current state since we're predicting the next action based on past states and actions
                past_states = torch.tensor(states[:-1], dtype=torch.float).unsqueeze(0)  # Shape: (1, seq_length)
                past_actions = torch.tensor(actions, dtype=torch.float).unsqueeze(0)     # Shape: (1, seq_length)
                past_rewards = torch.tensor(rewards, dtype=torch.float)                  # Shape: (seq_length,)
                returns_to_go = past_rewards.flip(0).cumsum(0).flip(0).unsqueeze(0)      # Shape: (1, seq_length)
                timesteps_tensor = torch.tensor(timesteps[:-1], dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)

                # Ensure inputs have correct dimensions
                past_states = past_states.unsqueeze(-1)       # Shape: (1, seq_length, 1)
                past_actions = past_actions.unsqueeze(-1)     # Shape: (1, seq_length, 1)
                returns_to_go = returns_to_go.unsqueeze(-1)   # Shape: (1, seq_length, 1)

                # Prepare input actions by shifting actions to the right
                input_actions = torch.zeros_like(past_actions)
                if past_actions.size(1) > 1:
                    input_actions[:, 1:, :] = past_actions[:, :-1, :]
                input_actions[:, 0, :] = 0  # Initial action is zero

                # **Step 3: Add print statements to verify shapes**
                print(f"past_states shape: {past_states.shape}")
                print(f"input_actions shape: {input_actions.shape}")
                print(f"returns_to_go shape: {returns_to_go.shape}")
                print(f"timesteps_tensor shape: {timesteps_tensor.shape}")

                # Call the model
                with torch.no_grad():
                    action_preds = model(
                        states=past_states,
                        actions=input_actions,
                        returns_to_go=returns_to_go,
                        timesteps=timesteps_tensor
                    )
                    action_pred = action_preds[:, -1, :]  # Get the last time step prediction
                    action_value = action_pred.item()
                    action_value = round(action_value)
                    action_value = max(0, min(action_value, env.action_space.n - 1))

                with torch.no_grad():
                    action_preds = model(
                        states=past_states,
                        actions=input_actions,
                        returns_to_go=returns_to_go,
                        timesteps=timesteps_tensor
                    )
                    action_pred = action_preds[:, -1, :]  # Get the last time step prediction
                    action_value = action_pred.item()
                    action_value = round(action_value)
                    action_value = max(0, min(action_value, env.action_space.n - 1))
            action = int(action_value)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if isinstance(obs, np.ndarray):
                obs = obs.item()
            states.append(obs)
            actions.append(action_value)
            rewards.append(reward)
            timesteps.append(step + 1)
            episode_rewards.append(reward)
            episode_info.append(info)
            step += 1
        all_rewards.append(sum(episode_rewards))
        all_info.append(episode_info)
    return all_rewards, all_info



class DecisionTransformerWithActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = DecisionTransformerModel(config)
        self.action_head = nn.Linear(config.hidden_size, config.act_dim)  # Predicts action values

    def forward(self, states, actions, returns_to_go, timesteps):
        # Verify input shapes
        print(f"states shape: {states.shape}")
        print(f"actions shape: {actions.shape}")
        print(f"returns_to_go shape: {returns_to_go.shape}")
        print(f"timesteps shape: {timesteps.shape}")

        transformer_outputs = self.transformer(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps
        )
        hidden_states = transformer_outputs.last_hidden_state  # Shape: (batch_size, seq_length * 3, hidden_size)

        # Positions of action tokens (every third token starting from index 2)
        batch_size, total_seq_length, _ = hidden_states.size()
        seq_length = states.size(1)
        action_token_positions = torch.arange(seq_length, device=states.device) * 3 + 2

        # Extract hidden states at action token positions
        action_hidden_states = hidden_states[:, action_token_positions, :]  # (batch_size, seq_length, hidden_size)

        action_preds = self.action_head(action_hidden_states)  # (batch_size, seq_length, act_dim)
        return action_preds