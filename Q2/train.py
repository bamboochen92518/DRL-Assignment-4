import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
from torch.distributions.normal import Normal
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Actor-Critic Network
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
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        value = self.critic(features)
        return dist, value

# PPO Agent
class PPO:
    def __init__(self, obs_size, act_size, lr, gamma, clip_eps, epochs, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_size, act_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        values = values + [next_value]
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, log_probs_old, returns, advantages):
        indices = np.random.permutation(len(states))
        
        for _ in range(self.epochs):
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_states = torch.FloatTensor(states[idx]).to(self.device)
                batch_actions = torch.FloatTensor(actions[idx]).to(self.device)
                batch_log_probs_old = torch.FloatTensor(log_probs_old[idx]).to(self.device)
                batch_returns = torch.FloatTensor(returns[idx]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[idx]).to(self.device)
                
                dist, value = self.model(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - value.squeeze()).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def evaluate(ppo, env, max_steps, num_episodes=5):
    rewards = []
    for _ in range(num_episodes):
        time_step = env.reset()
        obs = np.concatenate([time_step.observation['position'], time_step.observation['velocity']])
        episode_reward = 0
        
        for _ in range(max_steps):
            state = torch.FloatTensor(obs).to(ppo.device)
            dist, _ = ppo.model(state)
            action = dist.mean  # Deterministic action (mean of distribution)
            action_np = action.cpu().detach().numpy()
            time_step = env.step(action_np)
            obs = np.concatenate([time_step.observation['position'], time_step.observation['velocity']])
            episode_reward += time_step.reward
            
            if time_step.step_type == 2:
                break
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)

def main(args):
    # Initialize environment
    env = suite.load(domain_name="cartpole", task_name="balance")
    obs_size = env.observation_spec()['position'].shape[0] + env.observation_spec()['velocity'].shape[0]
    act_size = env.action_spec().shape[0]
    
    # Initialize PPO
    ppo = PPO(
        obs_size=obs_size,
        act_size=act_size,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Training parameters
    episode_rewards = []
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    
    for episode in range(args.max_episodes):
        time_step = env.reset()
        episode_reward = 0
        obs = np.concatenate([time_step.observation['position'], time_step.observation['velocity']])
        
        for step in range(args.max_steps):
            state = torch.FloatTensor(obs).to(ppo.device)
            dist, value = ppo.model(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
            action_np = action.cpu().detach().numpy()
            time_step = env.step(action_np)
            next_obs = np.concatenate([time_step.observation['position'], time_step.observation['velocity']])
            
            states.append(obs)
            actions.append(action_np)
            rewards.append(time_step.reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(0 if time_step.step_type != 2 else 1)
            
            obs = next_obs
            episode_reward += time_step.reward
            
            if time_step.step_type == 2:
                break
                
        episode_rewards.append(episode_reward)
        
        # Compute next value
        next_state = torch.FloatTensor(obs).to(ppo.device)
        _, next_value = ppo.model(next_state)
        
        # Compute returns and advantages
        advantages = ppo.compute_gae(rewards, values, next_value.item(), dones)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Update policy
        if len(states) >= args.update_freq:
            ppo.update(states, actions, log_probs, returns, advantages)
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        # Evaluation every 100 episodes
        if episode % 100 == 0 and episode > 0:
            mean_reward, std_reward = evaluate(ppo, env, args.max_steps)
            print(f"Evaluation at episode {episode}: Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
            # Save model if evaluation performance is good
            if mean_reward > 900:
                ppo.save_model(f"ppo_cartpole_{episode}.pth")
        
        # Early stopping
        if len(episode_rewards) > 50 and np.mean(episode_rewards[-50:]) > 990:
            print("Task solved!")
            break
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    ppo.save_model("models/ppo_cartpole_final.pth")
    print("Final model saved at models/ppo_cartpole_final.pth")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO for CartPole Balance")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--update_freq", type=int, default=2048, help="Update frequency in steps")
    
    args = parser.parse_args()
    main(args)