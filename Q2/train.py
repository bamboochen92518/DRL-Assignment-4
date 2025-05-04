import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
from tqdm import tqdm
import sys

# Add parent directory to sys.path for dmc import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

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
        states = np.array(states)
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        returns = np.array(returns)
        advantages = np.array(advantages)
        
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

def evaluate(ppo, env, num_episodes=5):
    episode_rewards = []
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        observation, info = env.reset(seed=np.random.randint(0, 1000000))
        
        episode_reward = 0
        done = False
        while not done:
            state = torch.FloatTensor(observation).to(ppo.device)
            dist, _ = ppo.model(state)
            action = dist.mean  # Deterministic action
            action_np = action.cpu().detach().numpy()
            observation, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    return episode_rewards

def main(args):
    # Initialize environment
    env = make_dmc_env("cartpole-balance", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    obs_size = env.observation_space.shape[0]  # Dynamically set (likely 5)
    act_size = env.action_space.shape[0]  # 1
    
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
    
    # Progress bar for episodes
    pbar = tqdm(range(args.max_episodes), desc="Training", unit="episode")
    for episode in pbar:
        observation, info = env.reset(seed=np.random.randint(0, 1000000))
        episode_reward = 0
        
        for step in range(args.max_steps):
            state = torch.FloatTensor(observation).to(ppo.device)
            dist, value = ppo.model(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
            action_np = action.cpu().detach().numpy()
            next_observation, reward, terminated, truncated, info = env.step(action_np)
            
            states.append(observation)
            actions.append(action_np)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(0 if not (terminated or truncated) else 1)
            
            observation = next_observation
            episode_reward += reward
            
            if terminated or truncated:
                break
                
        episode_rewards.append(episode_reward)
        
        # Compute next value
        next_state = torch.FloatTensor(observation).to(ppo.device)
        _, next_value = ppo.model(next_state)
        
        # Compute returns and advantages
        advantages = ppo.compute_gae(rewards, values, next_value.item(), dones)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Update policy
        if len(states) >= args.update_freq:
            ppo.update(states, actions, log_probs, returns, advantages)
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        # Update progress bar with current episode reward
        pbar.set_postfix({"Episode Reward": f"{episode_reward:.2f}"})
        
        # Evaluation every 100 episodes
        if episode % 100 == 0 and episode > 0:
            eval_rewards = evaluate(ppo, env, num_episodes=5)
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            tqdm.write(f"Evaluation at episode {episode}: Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
            # Save model if evaluation performance is good
            if mean_reward > 900:
                ppo.save_model(f"ppo_cartpole_{episode}.pth")
        
        # Early stopping
        if len(episode_rewards) > 50 and np.mean(episode_rewards[-50:]) > 990:
            tqdm.write("Task solved!")
            break
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    ppo.save_model("models/ppo_cartpole_final.pth")
    tqdm.write("Final model saved at models/ppo_cartpole_final.pth")
    
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