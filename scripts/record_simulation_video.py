"""
Simulation Video Recorder for BioFlight.
Renders closed-loop physics simulation of Drosophila flight in MuJoCo
and saves high-resolution MP4/WebP videos.
"""

import os
import sys
import shutil
import numpy as np
import imageio
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flybody.fly_envs import flight_imitation
from src.models.mlp_policy import MLPPolicy
from evaluation.run_evaluation import extract_features


def record_simulation():
    print("=" * 65)
    print(" BioFlight — Rendering Real-Time Closed-Loop MuJoCo Simulation")
    print("=" * 65)

    # Output directory
    videos_dir = os.path.join("videos")
    os.makedirs(videos_dir, exist_ok=True)

    # 1. Instantiate FlyBody environment without early termination distance limits
    env = flight_imitation(terminal_com_dist=float('inf'))
    env.reset()

    # 2. Load trained policy model
    ckpt_path = os.path.join("checkpoints", "imitation", "exp_001", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    policy = MLPPolicy(input_dim=28, action_dim=12)
    checkpoint = torch.load(ckpt_path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    # Mean and std stats from preprocessing
    meta_path = os.path.join("data", "processed", "metadata.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        mean_val = meta.get("mean") or meta.get("mean_X")
        std_val = meta.get("std") or meta.get("std_X")
        mean = np.array(mean_val, dtype=np.float32)
        std = np.array(std_val, dtype=np.float32)
    else:
        mean = np.zeros(28, dtype=np.float32)
        std = np.ones(28, dtype=np.float32)

    frames = []
    target_frames = 300  # 300 frames @ 30 FPS = 10.0 seconds of video playback
    render_interval = 5   # render every 5th physics step

    print(f"Simulating & rendering continuous 10-second video ({target_frames} frames @ 30 FPS)...")

    step = 0
    while len(frames) < target_frames:
        step += 1
        obs = extract_features(env, mean, std, lookahead_steps=5)
        action = policy.select_action(obs)
        time_step = env.step(action)

        if step % render_interval == 0:
            try:
                frame = env.physics.render(height=480, width=640, camera_id=0)
                frames.append(frame)
            except Exception as e:
                print(f"Render warning: {e}")
                break

    if len(frames) == 0:
        print("Error: No frames rendered.")
        sys.exit(1)

    # Save MP4 video
    mp4_path = os.path.join(videos_dir, "flight_simulation.mp4")
    webp_path = os.path.join(videos_dir, "flight_simulation.webp")

    imageio.mimsave(mp4_path, frames, fps=30)
    imageio.mimsave(webp_path, frames, fps=30)

    print(f"Rendered {len(frames)} frames successfully!")
    print(f"Saved MP4 video to:  {mp4_path}")
    print(f"Saved WebP video to: {webp_path}")

    # Copy to Artifact Directory for embedded visualization
    artifact_dir = r"C:\Users\Udit Jadhav\.gemini\antigravity-ide\brain\c4a3d6c4-0f25-48c4-8c03-457aa556395c"
    if os.path.exists(artifact_dir):
        artifact_mp4 = os.path.join(artifact_dir, "flight_simulation.mp4")
        artifact_webp = os.path.join(artifact_dir, "flight_simulation.webp")
        shutil.copy(mp4_path, artifact_mp4)
        shutil.copy(webp_path, artifact_webp)
        print(f"Copied video to artifact directory: {artifact_webp}")

    print("=" * 65)


if __name__ == "__main__":
    record_simulation()
