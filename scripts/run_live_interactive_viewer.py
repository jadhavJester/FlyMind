"""
Live Interactive Window Viewer for BioFlight.
Launches a real-time OpenCV desktop window displaying live FlyBody MuJoCo simulation.
Press 'q' or ESC to exit the live view.
"""

import os
import sys
import time
import numpy as np
import cv2
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flybody.fly_envs import flight_imitation
from src.models.mlp_policy import MLPPolicy
from evaluation.run_evaluation import extract_features


def run_live_viewer():
    print("=" * 65)
    print(" BioFlight — Opening Live Interactive Desktop Simulation Window")
    print(" Press 'q' or 'ESC' in the rendering window to close.")
    print("=" * 65)

    env = flight_imitation()
    env.reset()

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

    window_name = "BioFlight — Live MuJoCo FlyBody Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    step = 0
    while True:
        step += 1
        obs = extract_features(env, mean, std, lookahead_steps=5)
        action = policy.select_action(obs)
        env.step(action)

        # Render 640x480 frame
        rgb_frame = env.physics.render(height=480, width=640, camera_id=0)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Overlay telemetry HUD
        fly_pos = env.physics.data.qpos[:3]
        cv2.putText(bgr_frame, f"Step: {step:04d} | Pos (cm): X={fly_pos[0]:.2f}, Y={fly_pos[1]:.2f}, Z={fly_pos[2]:.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(bgr_frame, "BioFlight Neural Flight Policy (Closed-Loop)",
                    (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, bgr_frame)

        # 30 FPS playback (~33ms delay)
        key = cv2.waitKey(33) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break

    cv2.destroyAllWindows()
    print("Live viewer closed.")


if __name__ == "__main__":
    run_live_viewer()
