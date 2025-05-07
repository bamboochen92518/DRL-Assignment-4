import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def select_action(self, state):
        mu, _ = self.forward(state)
        action = torch.tanh(mu) * self.max_action
        return action

class Agent(object):
    """SAC Agent for Humanoid Walk.
    
    Expects observations as either a flat array of shape (67,) or a dictionary
    (from dm_control) that flattens to 67 dimensions.
    """
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 67  # Based on Humanoid Walk observation space
        self.action_dim = 21  # Based on Humanoid Walk action space
        self.hidden_dim = 256
        self.max_action = 1.0

        # Initialize actor
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action).to(self.device)
        
        # Load trained model
        try:
            self.actor.load_state_dict(torch.load("models/actor.pth"))
            self.actor.eval()
        except FileNotFoundError:
            raise FileNotFoundError("Model file 'models/actor.pth' not found. Please train the model first.")
        
    def act(self, observation):
        # Handle dictionary observation (from dm_control) or flat array
        if isinstance(observation, dict):
            observation = np.concatenate([obs.flatten() for obs in observation.values()])
        
        # Verify observation shape
        if observation.shape != (self.state_dim,):
            raise ValueError(f"Expected observation shape ({self.state_dim},), got {observation.shape}")
        
        # Convert observation to tensor
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action
        with torch.no_grad():
            action = self.actor.select_action(state)
        
        # Convert to numpy and ensure it fits action space
        action = action.cpu().numpy().flatten()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action