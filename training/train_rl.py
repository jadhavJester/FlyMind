"""
Reinforcement Learning Training Script for BioFlight (Phase 8).
Trains an Actor-Critic PPO policy to stabilize Drosophila flight at target altitude.
"""

import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.flight_stabilization_env import FlightStabilizationEnv


class ActorCritic(nn.Module):
    """Gaussian Policy Actor-Critic Network for Continuous Control."""

    def __init__(self, obs_dim: int = 16, action_dim: int = 12, hidden_dims: list = [128, 128]):
        super().__init__()
        
        # Actor network
        actor_layers = []
        prev = obs_dim
        for h in hidden_dims:
            actor_layers.append(nn.Linear(prev, h))
            actor_layers.append(nn.Tanh())
            prev = h
        actor_layers.append(nn.Linear(prev, action_dim))
        self.actor_mean = nn.Sequential(*actor_layers)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        # Critic network
        critic_layers = []
        prev = obs_dim
        for h in hidden_dims:
            critic_layers.append(nn.Linear(prev, h))
            critic_layers.append(nn.Tanh())
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state):
        mean = self.actor_mean(state)
        std = torch.exp(self.actor_logstd)
        value = self.critic(state)
        return mean, std, value

    def act(self, state):
        with torch.no_grad():
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = torch.clamp(action, -1.0, 1.0)
            return action.numpy()[0], log_prob.item(), value.item()

    def evaluate(self, states, actions):
        mean, std, values = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy


def train_rl():
    with open("configs/rl.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"].get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    results_dir = cfg["experiment"]["results_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    env = FlightStabilizationEnv(
        target_pos=np.array(cfg["environment"]["target_pos"], dtype=np.float32),
        target_vel=np.array(cfg["environment"]["target_vel"], dtype=np.float32),
        max_steps=cfg["environment"]["max_steps"],
        action_scale=cfg["environment"]["action_scale"],
        reward_weights=cfg["environment"]["reward_weights"]
    )

    model = ActorCritic(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dims=cfg["algorithm"]["actor_dims"]
    )

    optimizer = Adam(model.parameters(), lr=cfg["algorithm"]["learning_rate"])

    total_timesteps = cfg["algorithm"]["total_timesteps"]
    clip_range = cfg["algorithm"]["clip_range"]
    gamma = cfg["algorithm"]["gamma"]
    n_epochs = cfg["algorithm"]["n_epochs"]

    print("=" * 65)
    print(" BioFlight — Phase 8 RL (PPO) Flight Stabilization Training")
    print(f" Target Position: {cfg['environment']['target_pos']} cm | Total Timesteps: {total_timesteps:,}")
    print("=" * 65)

    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    episode_returns = []
    episode_lengths = []

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_steps = 0
    total_steps = 0
    ep_count = 0

    best_reward = -float("inf")

    start_time = time.time()

    while total_steps < total_timesteps:
        total_steps += 1
        ep_steps += 1

        state_t = torch.from_numpy(obs).float().unsqueeze(0)
        action, log_prob, val = model.act(state_t)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(val)
        dones.append(done)

        obs = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            episode_returns.append(ep_reward)
            episode_lengths.append(ep_steps)

            if ep_count % 10 == 0:
                avg_ret = np.mean(episode_returns[-10:])
                avg_len = np.mean(episode_lengths[-10:])
                print(f"Step [{total_steps:6d}/{total_timesteps:6d}] | Ep {ep_count:3d} | Avg Return: {avg_ret:6.2f} | Avg Steps: {avg_len:5.1f}")
                if avg_ret > best_reward:
                    best_reward = avg_ret
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_policy.pt"))

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0

        # PPO Update Buffer
        if len(states) >= 1024:
            # Compute Returns and Advantages
            returns = []
            discounted = 0
            for r, d in zip(reversed(rewards), reversed(dones)):
                if d:
                    discounted = 0
                discounted = r + gamma * discounted
                returns.insert(0, discounted)

            states_t = torch.tensor(np.array(states), dtype=torch.float32)
            actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
            old_log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32)
            returns_t = torch.tensor(np.array(returns), dtype=torch.float32)
            values_t = torch.tensor(np.array(values), dtype=torch.float32)

            advantages_t = returns_t - values_t
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            for _ in range(n_epochs):
                new_log_probs, new_values, entropy = model.evaluate(states_t, actions_t)
                ratios = torch.exp(new_log_probs - old_log_probs_t)

                surr1 = ratios * advantages_t
                surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages_t

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_values, returns_t)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    print("-" * 65)
    print(f"PPO Training Complete in {time.time() - start_time:.1f}s | Best 10-Ep Avg Return: {best_reward:.2f}")

    # Plot & Save Returns
    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(episode_returns, color='#2A9D8F', alpha=0.4, label='Episode Return')
    smooth = np.convolve(episode_returns, np.ones(10)/10, mode='valid')
    plt.plot(np.arange(9, len(episode_returns)), smooth, color='#2A9D8F', linewidth=2, label='10-Ep Moving Avg')
    plt.title('BioFlight Phase 8 — PPO Flight Stabilization Training Curves')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rl_training_curves.png"))
    plt.close()

    print(f"Saved RL policy checkpoint to {os.path.join(ckpt_dir, 'best_policy.pt')}")
    print(f"Saved training plot to {os.path.join(results_dir, 'rl_training_curves.png')}")


if __name__ == "__main__":
    train_rl()
