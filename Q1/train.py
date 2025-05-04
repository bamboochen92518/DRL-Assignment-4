import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy
import argparse
import os

# Actor Network
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

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, args):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(args.buffer_capacity)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
    
    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise:
            action = (action + np.random.normal(0, 0.1, size=action.shape)).clip(-self.max_action, self.max_action)
        return action
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(device)
        
        # Critic update
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()
        
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)

# Evaluation Function
def evaluate_agent(agent, env, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        step = 0
        while not done and step < args.max_steps:
            action = agent.select_action(state, noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            step += 1
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# Training Loop
def train_ddpg(args):
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = DDPG(state_dim, action_dim, max_action, args)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ddpg_pendulum.pth'
    
    for episode in range(args.episodes):
        state = env.reset()[0]
        episode_reward = 0
        step = 0
        
        while step < args.max_steps:
            action = agent.select_action(state, noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        print(f"Episode {episode+1}/{args.episodes}, Reward: {episode_reward:.2f}, Steps: {step}")
        
        # Evaluate every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = evaluate_agent(agent, env)
            print(f"Evaluation at Episode {episode+1}: Average Reward = {avg_reward:.2f}")
    
    # Save the model
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG for Pendulum-v1')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--buffer_capacity', type=int, default=1000000, help='Replay buffer capacity')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ddpg(args)