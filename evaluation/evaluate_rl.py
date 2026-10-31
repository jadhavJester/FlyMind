"""
Evaluation script for Phase 8 RL Flight Stabilization Policy.
Tests trained Actor-Critic policy in live FlyBody MuJoCo simulation across 10 evaluation episodes.
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.flight_stabilization_env import FlightStabilizationEnv
from training.train_rl import ActorCritic


def evaluate_rl():
    with open("configs/rl.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    results_dir = cfg["experiment"]["results_dir"]
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

    ckpt_path = os.path.join(ckpt_dir, "best_policy.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    num_episodes = 10
    total_returns = []
    episode_steps = []
    pos_errors = []
    crashes = 0

    print("=" * 65)
    print(" BioFlight — Phase 8 RL Closed-Loop Hover Stabilization Evaluation")
    print("=" * 65)

    all_positions = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + 100)
        ep_ret = 0.0
        steps = 0
        ep_pos_err = []
        ep_positions = []

        while True:
            steps += 1
            state_t = torch.from_numpy(obs).float().unsqueeze(0)
            action, _, _ = model.act(state_t)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_pos_err.append(info["pos_err_cm"])
            ep_positions.append(env.physics.data.qpos[:3].copy())

            if terminated or truncated:
                if info["crashed"]:
                    crashes += 1
                break

        total_returns.append(ep_ret)
        episode_steps.append(steps)
        pos_errors.append(np.mean(ep_pos_err))
        all_positions.append(np.array(ep_positions))

        print(f" Episode {ep+1:2d}/{num_episodes} | Steps: {steps:3d} | Return: {ep_ret:6.2f} | Pos Error: {np.mean(ep_pos_err):.3f} cm | Crashed: {info['crashed']}")

    survival_rate = (1.0 - (crashes / num_episodes)) * 100.0

    print("-" * 65)
    print(f"Survival Rate:        {survival_rate:.1f}%")
    print(f"Average Episode Return: {np.mean(total_returns):.2f}")
    print(f"Average Position Error: {np.mean(pos_errors):.3f} cm ({np.mean(pos_errors)*10:.2f} mm)")
    print("=" * 65)

    # Plot trajectories
    fig = plt.figure(figsize=(9, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    for pos_arr in all_positions:
        ax.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], color='#2A9D8F', alpha=0.6)
    
    target = cfg["environment"]["target_pos"]
    ax.scatter([target[0]], [target[1]], [target[2]], color='#E63946', s=100, label='Target Hover Position')
    ax.set_title('BioFlight Phase 8 — RL Hover Stabilization 3D Rollouts')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rl_hover_3d.png"))
    plt.close()

    metrics = {
        "num_episodes": num_episodes,
        "survival_rate_percent": survival_rate,
        "mean_return": float(np.mean(total_returns)),
        "mean_pos_error_cm": float(np.mean(pos_errors)),
        "mean_steps": float(np.mean(episode_steps))
    }

    with open(os.path.join(results_dir, "rl_evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved RL evaluation metrics to {os.path.join(results_dir, 'rl_evaluation.json')}")
    print(f"Saved 3D plot to {os.path.join(results_dir, 'rl_hover_3d.png')}")


if __name__ == "__main__":
    evaluate_rl()
