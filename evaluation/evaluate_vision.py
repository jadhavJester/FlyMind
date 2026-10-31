"""
Comparative Evaluation Script for Phase 9 (Vision-Guided Flight).
Compares Architecture A (End-to-End Direct Control) vs Architecture B (Hierarchical Control).
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

from src.environments.vision_target_env import VisionTargetEnv
from src.models.mlp_policy import MLPPolicy
from src.models.vision_policy import DirectVisionPolicy, HierarchicalVisionPolicy


def evaluate_vision():
    with open("configs/vision.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["experiment"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    env = VisionTargetEnv(
        image_shape=tuple(cfg["environment"]["image_shape"]),
        max_steps=cfg["environment"]["max_steps"],
        action_scale=cfg["environment"]["action_scale"]
    )

    # Instantiate Architecture A
    arch_a = DirectVisionPolicy(action_dim=12)
    arch_a.eval()

    # Instantiate Architecture B (Hierarchical) with low-level pretrained policy
    low_level = MLPPolicy(input_dim=28, action_dim=12)
    ckpt_path = cfg["evaluation"]["low_level_checkpoint"]
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        low_level.load_state_dict(state_dict)
    low_level.eval()

    arch_b = HierarchicalVisionPolicy(low_level_policy=low_level)
    arch_b.eval()

    num_episodes = cfg["evaluation"]["num_episodes"]

    print("=" * 70)
    print(" BioFlight — Phase 9 Vision-Guided Flight Architecture Comparison")
    print(" Architecture A: Direct End-to-End Control (Image -> CNN -> Actions)")
    print(" Architecture B: Hierarchical Control (Image -> CNN -> Command -> Low-Level MLP)")
    print("=" * 70)

    def run_eval(policy_model, is_hierarchical=False):
        distances = []
        target_reached_count = 0
        survival_count = 0

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep + 42)
            ep_dist = []

            while True:
                img_t = torch.from_numpy(obs["image"]).unsqueeze(0)
                kin_t = torch.from_numpy(obs["kinematic"]).unsqueeze(0)

                with torch.no_grad():
                    if is_hierarchical:
                        action = policy_model(img_t, kin_t).squeeze(0).numpy()
                    else:
                        action = policy_model(img_t).squeeze(0).numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_dist.append(info["dist_to_target_cm"])

                if terminated or truncated:
                    if info["target_reached"]:
                        target_reached_count += 1
                    if not info["crashed"]:
                        survival_count += 1
                    break

            distances.append(np.mean(ep_dist))

        return {
            "mean_dist_cm": float(np.mean(distances)),
            "interception_rate_percent": (target_reached_count / num_episodes) * 100.0,
            "survival_rate_percent": (survival_count / num_episodes) * 100.0
        }

    res_a = run_eval(arch_a, is_hierarchical=False)
    res_b = run_eval(arch_b, is_hierarchical=True)

    print(f"Architecture A (Direct End-to-End):")
    print(f"  - Mean Target Distance:   {res_a['mean_dist_cm']:.2f} cm")
    print(f"  - Interception Success:    {res_a['interception_rate_percent']:.1f}%")
    print(f"  - Flight Survival Rate:    {res_a['survival_rate_percent']:.1f}%\n")

    print(f"Architecture B (Hierarchical Control):")
    print(f"  - Mean Target Distance:   {res_b['mean_dist_cm']:.2f} cm")
    print(f"  - Interception Success:    {res_b['interception_rate_percent']:.1f}%")
    print(f"  - Flight Survival Rate:    {res_b['survival_rate_percent']:.1f}%")
    print("=" * 70)

    # Save comparative bar chart
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    
    archs = ['Arch A\n(Direct)', 'Arch B\n(Hierarchical)']
    survivals = [res_a['survival_rate_percent'], res_b['survival_rate_percent']]
    distances = [res_a['mean_dist_cm'], res_b['mean_dist_cm']]

    ax[0].bar(archs, survivals, color=['#E63946', '#2A9D8F'], width=0.4)
    ax[0].set_ylabel('Survival / Stability Rate (%)')
    ax[0].set_ylim(0, 105)
    ax[0].set_title('Flight Stability Comparison')
    ax[0].grid(True, linestyle='--', alpha=0.5)

    ax[1].bar(archs, distances, color=['#E63946', '#2A9D8F'], width=0.4)
    ax[1].set_ylabel('Mean Distance to Target (cm)')
    ax[1].set_title('Visual Target Tracking Precision')
    ax[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('BioFlight Phase 9 — Vision-Guided Control Architecture Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "vision_architecture_comparison.png"))
    plt.close()

    report = {
        "architecture_a_direct": res_a,
        "architecture_b_hierarchical": res_b
    }

    with open(os.path.join(results_dir, "vision_evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved JSON report to {os.path.join(results_dir, 'vision_evaluation.json')}")
    print(f"Saved comparative chart to {os.path.join(results_dir, 'vision_architecture_comparison.png')}")


if __name__ == "__main__":
    evaluate_vision()
