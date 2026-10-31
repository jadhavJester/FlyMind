"""
3D Room Roaming Script for BioFlight.
Simulates and renders fruit fly roaming freely around a 3D enclosed room space.
Generates video animation files and interactive desktop telemetry display.
"""

import os
import sys
import shutil
import numpy as np
import imageio
import cv2
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.room_roaming_env import RoomRoamingEnv
from src.models.mlp_policy import MLPPolicy


def run_room_roaming():
    print("=" * 65)
    print(" BioFlight — 3D Enclosed Room Free-Roaming Simulation")
    print("=" * 65)

    videos_dir = os.path.join("videos")
    os.makedirs(videos_dir, exist_ok=True)

    # 1. Instantiate 3D Room Environment
    env = RoomRoamingEnv(room_size=(12.0, 12.0, 6.0))
    obs, _ = env.reset(seed=123)

    # 2. Load trained neural policy
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

    frames = []
    num_frames = 300  # 10.0 seconds of room roaming video @ 30 FPS
    render_interval = 5

    window_name = "BioFlight — 3D Enclosed Room Free-Roaming Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    waypoints_visited = 0
    step = 0

    while len(frames) < num_frames:
        step += 1
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if info["waypoint_reached"]:
            waypoints_visited += 1

        if step % render_interval == 0:
            try:
                frame = env.physics.render(height=480, width=640, camera_id=0)
                
                # Draw visual room telemetry overlay on video frames
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                fly_p = info["fly_pos_cm"]
                wpt_p = info["target_waypoint_cm"]

                cv2.putText(bgr, f"3D ROOM ROAMING — Step: {step:04d} | Waypoints Visited: {waypoints_visited}",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(bgr, f"Fly Pos: [{fly_p[0]:.1f}, {fly_p[1]:.1f}, {fly_p[2]:.1f}] cm | Target: [{wpt_p[0]:.1f}, {wpt_p[1]:.1f}, {wpt_p[2]:.1f}] cm",
                            (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Show live window
                cv2.imshow(window_name, bgr)
                cv2.waitKey(1)

                rgb_annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb_annotated)
            except Exception as e:
                print(f"Render warning: {e}")
                break

        if terminated:
            obs, _ = env.reset()

    cv2.destroyAllWindows()

    # Save output videos
    mp4_path = os.path.join(videos_dir, "fly_room_roaming.mp4")
    webp_path = os.path.join(videos_dir, "fly_room_roaming.webp")

    imageio.mimsave(mp4_path, frames, fps=30)
    imageio.mimsave(webp_path, frames, fps=30)

    print(f"Rendered {len(frames)} frames cleanly!")
    print(f"Visited {waypoints_visited} 3D room waypoints during roaming!")
    print(f"Saved MP4 video to:  {mp4_path}")
    print(f"Saved WebP video to: {webp_path}")

    # Copy to artifact directory for embedded viewing
    artifact_dir = r"C:\Users\Udit Jadhav\.gemini\antigravity-ide\brain\c4a3d6c4-0f25-48c4-8c03-457aa556395c"
    if os.path.exists(artifact_dir):
        shutil.copy(webp_path, os.path.join(artifact_dir, "fly_room_roaming.webp"))
        shutil.copy(mp4_path, os.path.join(artifact_dir, "fly_room_roaming.mp4"))

    print("=" * 65)


if __name__ == "__main__":
    run_room_roaming()
