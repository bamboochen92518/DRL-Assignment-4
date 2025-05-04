import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

class Agent(object):
    """DDPG Agent for Pendulum-v1."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        
        # Initialize actor network
        state_dim = 3  # Pendulum-v1 state: [cos(theta), sin(theta), theta_dot]
        action_dim = 1  # Pendulum-v1 action: torque
        max_action = 2.0  # Max action value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        
        # Load trained model
        model_path = 'models/ddpg_pendulum.pth'
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.eval()
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    def act(self, observation):
        # Convert observation to tensor
        state = torch.FloatTensor(observation).reshape(1, -1).to(self.device)
        # Get action from actor (no exploration noise for inference)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        # Ensure action is within bounds
        action = np.clip(action, -2.0, 2.0)
        return action