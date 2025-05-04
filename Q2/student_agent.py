import gymnasium
import numpy as np
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Linear(64, act_size)
        self.actor_logstd = nn.Parameter(torch.zeros(act_size))
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.shared(x)
        mean = self.actor_mean(features)
        return mean

class Agent(object):
    """Agent that uses a trained PPO model for action selection."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        obs_size = 4  # CartPole Balance: [position, velocity] = 2 + 2
        act_size = 1
        self.model = ActorCritic(obs_size, act_size).to(self.device)
        
        # Load trained model
        try:
            self.model.load_state_dict(torch.load("models/ppo_cartpole_final.pth", map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            print("Warning: Model file not found. Using untrained model.")
        
    def act(self, observation):
        # Ensure observation is a numpy array and has correct shape
        obs = np.array(observation, dtype=np.float32)
        if obs.shape != (4,):
            obs = obs.flatten()[:4]
        
        # Convert to tensor and get action
        state = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action = self.model(state)
        
        # Ensure action is within action space
        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)
        
        return action_np