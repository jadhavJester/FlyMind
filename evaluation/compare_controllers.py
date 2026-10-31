"""
Comparative Research Evaluation Script for Phase 10 (Bio-Inspired UAV Control Transfer).
Compares Conventional PID vs Bio-Inspired Transfer Controller under baseline and wind gust disturbances.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.uav.quadrotor_env import QuadrotorEnv
from src.controllers.uav_controllers import ConventionalPIDQuadrotorController, BioInspiredTransferController


def compare_uav_controllers():
    print("=" * 70)
    print(" BioFlight — Phase 10 UAV Control Transfer Research Comparison")
    print(" Conventional PID vs Bio-Inspired Transfer Controller")
    print("=" * 70)

    results_dir = os.path.join("results", "uav", "exp_001")
    os.makedirs(results_dir, exist_ok=True)

    dt = 0.001
    steps = 1000

    pid_ctrl = ConventionalPIDQuadrotorController()
    bio_ctrl = BioInspiredTransferController()

    target_trajectory = []
    for t in range(steps):
        # Evasive Saccade Reference Trajectory (Banked Turn maneuver)
        x_target = 0.5 * np.sin(0.005 * t)
        y_target = 0.5 * (1.0 - np.cos(0.005 * t))
        z_target = 1.0 + 0.1 * np.sin(0.01 * t)
        target_trajectory.append(np.array([x_target, y_target, z_target], dtype=np.float32))

    def evaluate_controller(controller, wind_std=0.0):
        env = QuadrotorEnv(dt=dt, max_steps=steps, wind_std=wind_std)
        state, _ = env.reset(seed=42)
        
        pos_errors = []
        efforts = []
        positions = []
        roll_angles = []

        for t in range(steps):
            target_pos = target_trajectory[t]
            thrusts = controller.compute_action(state, target_pos)

            state, reward, terminated, truncated, info = env.step(thrusts)

            pos = state[:3]
            pos_err = np.linalg.norm(target_pos - pos)

            pos_errors.append(pos_err)
            efforts.append(np.mean(np.square(thrusts)))
            positions.append(pos.copy())
            roll_angles.append(np.degrees(abs(state[6])))

            if terminated:
                break

        return {
            "pos_rmse_m": float(np.sqrt(np.mean(np.square(pos_errors)))),
            "pos_mae_m": float(np.mean(pos_errors)),
            "control_effort": float(np.mean(efforts)),
            "max_roll_deg": float(np.max(roll_angles)),
            "positions": np.array(positions),
            "survival_steps": len(positions)
        }

    # 1. Baseline Evaluation (No Wind)
    res_pid_clean = evaluate_controller(pid_ctrl, wind_std=0.0)
    res_bio_clean = evaluate_controller(bio_ctrl, wind_std=0.0)

    # 2. Disturbance Evaluation (Continuous Wind Gusts std = 1.5 m/s)
    res_pid_wind = evaluate_controller(pid_ctrl, wind_std=1.5)
    res_bio_wind = evaluate_controller(bio_ctrl, wind_std=1.5)

    print("\n--- Baseline Performance (No Wind) ---")
    print(f"Conventional PID: RMSE = {res_pid_clean['pos_rmse_m']*100:.2f} cm | Effort = {res_pid_clean['control_effort']:.2f} | Peak Roll = {res_pid_clean['max_roll_deg']:.1f}°")
    print(f"Bio-Inspired:     RMSE = {res_bio_clean['pos_rmse_m']*100:.2f} cm | Effort = {res_bio_clean['control_effort']:.2f} | Peak Roll = {res_bio_clean['max_roll_deg']:.1f}°")

    print("\n--- Disturbance Rejection (Wind Gusts @ 1.5 m/s) ---")
    print(f"Conventional PID: RMSE = {res_pid_wind['pos_rmse_m']*100:.2f} cm | Effort = {res_pid_wind['control_effort']:.2f} | Peak Roll = {res_pid_wind['max_roll_deg']:.1f}°")
    print(f"Bio-Inspired:     RMSE = {res_bio_wind['pos_rmse_m']*100:.2f} cm | Effort = {res_bio_wind['control_effort']:.2f} | Peak Roll = {res_bio_wind['max_roll_deg']:.1f}°")
    print("=" * 70)

    # Save 3D Comparative Trajectory Plot
    fig = plt.figure(figsize=(9, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    target_arr = np.array(target_trajectory)
    ax.plot(target_arr[:, 0], target_arr[:, 1], target_arr[:, 2], 'k--', label='Reference Evasion Path', linewidth=1.5)

    pos_pid = res_pid_clean['positions']
    ax.plot(pos_pid[:, 0], pos_pid[:, 1], pos_pid[:, 2], color='#E63946', label='Conventional PID', linewidth=2.0)

    pos_bio = res_bio_clean['positions']
    ax.plot(pos_bio[:, 0], pos_bio[:, 1], pos_bio[:, 2], color='#2A9D8F', label='Bio-Inspired Controller', linewidth=2.0)

    ax.set_title('BioFlight Phase 10 — Quadrotor UAV Evasion Saccade Tracking')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "uav_trajectory_comparison.png"))
    plt.close()

    # Save Summary Report JSON
    summary = {
        "baseline_no_wind": {
            "conventional_pid": {
                "pos_rmse_cm": res_pid_clean['pos_rmse_m'] * 100,
                "pos_mae_cm": res_pid_clean['pos_mae_m'] * 100,
                "control_effort": res_pid_clean['control_effort'],
                "max_roll_deg": res_pid_clean['max_roll_deg']
            },
            "bio_inspired_transfer": {
                "pos_rmse_cm": res_bio_clean['pos_rmse_m'] * 100,
                "pos_mae_cm": res_bio_clean['pos_mae_m'] * 100,
                "control_effort": res_bio_clean['control_effort'],
                "max_roll_deg": res_bio_clean['max_roll_deg']
            }
        },
        "wind_gust_disturbance": {
            "conventional_pid": {
                "pos_rmse_cm": res_pid_wind['pos_rmse_m'] * 100,
                "pos_mae_cm": res_pid_wind['pos_mae_m'] * 100,
                "control_effort": res_pid_wind['control_effort'],
                "max_roll_deg": res_pid_wind['max_roll_deg']
            },
            "bio_inspired_transfer": {
                "pos_rmse_cm": res_bio_wind['pos_rmse_m'] * 100,
                "pos_mae_cm": res_bio_wind['pos_mae_m'] * 100,
                "control_effort": res_bio_wind['control_effort'],
                "max_roll_deg": res_bio_wind['max_roll_deg']
            }
        }
    }

    with open(os.path.join(results_dir, "uav_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved research report to {os.path.join(results_dir, 'uav_comparison.json')}")
    print(f"Saved 3D plot to {os.path.join(results_dir, 'uav_trajectory_comparison.png')}")


if __name__ == "__main__":
    compare_uav_controllers()
