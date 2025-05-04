import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dm_control import suite
from collections import deque
import random
from tqdm import tqdm
import os

# Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.action[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.reward[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_state[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.done[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx]
        )

# Actor Network
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

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# SAC Agent
class SAC:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # Environment setup
        self.env = suite.load(args.domain_name, args.task_name)
        self.state_dim = sum(np.prod(spec.shape) for spec in self.env.observation_spec().values())
        self.action_dim = np.prod(self.env.action_spec().shape)
        self.max_action = float(self.env.action_spec().maximum[0])

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, args.hidden_dim, self.max_action).to(self.device)
        self.critic_1 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic_2 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        # Copy weights to target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=args.lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=args.lr_critic)

        # Entropy tuning
        self.target_entropy = -self.action_dim if args.target_entropy is None else args.target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr_alpha)
        self.alpha = self.log_alpha.exp()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, args.replay_buffer_size, self.device)

    def get_state(self, observation):
        return torch.tensor(np.concatenate([obs.flatten() for obs in observation.values()]), dtype=torch.float32, device=self.device)

    def update(self):
        if self.replay_buffer.size < self.args.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.args.batch_size)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic_1(next_states, next_actions)
            q2_next = self.target_critic_2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.args.gamma * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def evaluate(self):
        self.actor.eval()
        eval_rewards = []
        for _ in range(10):  # Evaluate over 10 episodes
            time_step = self.env.reset()
            state = self.get_state(time_step.observation)
            episode_reward = 0
            for _ in range(self.args.max_steps):
                with torch.no_grad():
                    action, _ = self.actor.sample(state.unsqueeze(0))
                    action = action.cpu().numpy().flatten()
                time_step = self.env.step(action)
                state = self.get_state(time_step.observation)
                episode_reward += time_step.reward
                if time_step.last():
                    break
            eval_rewards.append(episode_reward)
        self.actor.train()
        return np.mean(eval_rewards)

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(), "models/actor.pth")
        torch.save(self.critic_1.state_dict(), "models/critic_1.pth")
        torch.save(self.critic_2.state_dict(), "models/critic_2.pth")
        print("Model saved to models/")

    def train(self):
        episode_rewards = deque(maxlen=100)
        progress_bar = tqdm(range(self.args.max_episodes), desc="Training", unit="episode")

        for episode in progress_bar:
            time_step = self.env.reset()
            state = self.get_state(time_step.observation)
            episode_reward = 0
            step = 0

            for step in range(self.args.max_steps):
                with torch.no_grad():
                    action, _ = self.actor.sample(state.unsqueeze(0))
                    action = action.cpu().numpy().flatten()

                time_step = self.env.step(action)
                next_state = self.get_state(time_step.observation)
                reward = time_step.reward
                done = 1.0 if time_step.last() else 0.0

                self.replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
                state = next_state
                episode_reward += reward

                for _ in range(self.args.update_steps):
                    self.update()

                if done:
                    break

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards)
            progress_bar.set_postfix({"Reward": f"{episode_reward:.2f}", "Avg Reward": f"{avg_reward:.2f}"})

            # Evaluate every 100 episodes
            if (episode + 1) % 100 == 0:
                eval_reward = self.evaluate()
                print(f"Episode {episode + 1}/{self.args.max_episodes}, Eval Reward: {eval_reward:.2f}")

            if avg_reward > 900:  # Arbitrary threshold for Humanoid Walk
                print("Task solved!")
                break

        self.save_model()

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on DeepMind Control Suite - Humanoid Walk")
    parser.add_argument("--domain_name", type=str, default="humanoid", help="Domain name for DMC")
    parser.add_argument("--task_name", type=str, default="walk", help="Task name for DMC")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--lr_alpha", type=float, default=3e-4, help="Alpha learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--replay_buffer_size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--update_steps", type=int, default=1, help="Number of updates per step")
    parser.add_argument("--target_entropy", type=float, default=None, help="Target entropy (default: -action_dim)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    agent = SAC(args)
    agent.train()