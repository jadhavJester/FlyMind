"""
Milestone M4A — Empirical Flight Envelope Characterization.

Systematically measures the safe flight envelope of the existing FlyBody policy:
- Distances: 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00 cm
- Headings: 0, 10, 20, 30, 35, 40, 45 degrees
- Evaluates altitude loss, touchdown rate, tracking error, and contact success.
- Produces CSV, JSON, summary plots, and representative simulation video.
"""

import os
import sys
import json
import csv
import time
from typing import Dict, List, Any, Tuple
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import flybody
from flybody import fly_envs
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from src.agents.oracle_selector import FlightPolicyController
from src.data.flight_dataset import quat_to_euler


def run_single_episode(
    env,
    controller: FlightPolicyController,
    distance_cm: float,
    heading_deg: float,
    record_frames: bool = False
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """Runs a single deterministic flight envelope trial."""
    dt = env.control_timestep()
    heading_rad = np.deg2rad(heading_deg)

    # Target position relative to origin (0, 0, 1.037)
    target_pos = np.array([
        distance_cm * np.cos(heading_rad),
        distance_cm * np.sin(heading_rad),
        1.037
    ], dtype=np.float32)

    # Compute trajectory duration based on distance at cruise speed 20 cm/s
    speed = 20.0  # cm/s
    flight_time = distance_cm / speed
    n_traj_steps = max(200, int(flight_time / dt) + 150)
    max_steps = n_traj_steps + 150

    qp, qv = constant_speed_trajectory(
        n_steps=n_traj_steps,
        speed=speed,
        init_pos=(0.0, 0.0, 1.037),
        init_heading=heading_rad,
        body_rot_angle_y=-47.5,
        control_timestep=dt
    )
    env.task._traj_generator.set_next_trajectory(qp, qv)
    env.reset()

    physics = env.physics
    initial_pos = physics.named.data.xpos['walker/thorax'].copy()
    initial_altitude = float(initial_pos[2])

    min_altitude = initial_altitude
    min_target_dist = float(np.linalg.norm(initial_pos - target_pos))
    max_heading_dev = 0.0

    success = False
    touchdown = False
    collision = False
    timeout = False
    step_count = 0
    frames = []

    com_qpos = getattr(env.task._traj_generator, '_com_qpos', None)

    for step in range(max_steps):
        step_count += 1
        fly_qvel = physics.data.qvel[:6]
        fly_lin_vel = fly_qvel[:3].copy()
        fly_ang_vel = fly_qvel[3:6].copy()

        fly_qpos = physics.data.qpos[:7]
        fly_pos = physics.named.data.xpos['walker/thorax'].copy()
        fly_quat = fly_qpos[3:7].copy()
        fly_euler = quat_to_euler(fly_quat)

        cur_alt = float(fly_pos[2])
        if cur_alt < min_altitude:
            min_altitude = cur_alt

        target_dist = float(np.linalg.norm(fly_pos - target_pos))
        if target_dist < min_target_dist:
            min_target_dist = target_dist

        # Compute instant heading deviation
        fly_heading = float(np.arctan2(fly_lin_vel[1], fly_lin_vel[0])) if np.linalg.norm(fly_lin_vel[:2]) > 1.0 else heading_rad
        heading_dev = abs(float(np.rad2deg(fly_heading - heading_rad)))
        if heading_dev > 180.0:
            heading_dev = 360.0 - heading_dev
        if heading_dev > max_heading_dev:
            max_heading_dev = heading_dev

        # Lookahead vectors (H=5)
        future_disp = []
        for k in range(1, 6):
            t_step = step + k
            if com_qpos is not None and isinstance(com_qpos, np.ndarray) and t_step < com_qpos.shape[0]:
                ref_pt = com_qpos[t_step, :3]
            else:
                ref_pt = target_pos
            future_disp.extend(ref_pt - fly_pos)

        raw_feature = np.concatenate([
            fly_lin_vel,
            fly_ang_vel,
            fly_quat,
            fly_euler,
            np.array(future_disp, dtype=np.float32)
        ]).astype(np.float32)

        action = controller.policy.select_action((raw_feature - controller.mean) / controller.std)
        env.step(action)

        if record_frames and step % 4 == 0:
            frame = physics.render(camera_id='walker/track1', height=240, width=320)
            frames.append(frame)

        # Contact criterion: 0.15 cm
        if target_dist <= 0.15:
            success = True
            break

        # Touchdown criterion: altitude < 0.20 cm
        if cur_alt < 0.20:
            touchdown = True
            break

        # Arena boundaries
        if not (-2.0 <= fly_pos[0] <= 10.0 and -6.0 <= fly_pos[1] <= 6.0 and fly_pos[2] <= 6.0):
            collision = True
            break

    if not success and not touchdown and not collision:
        timeout = True

    final_pos = physics.named.data.xpos['walker/thorax'].copy()
    final_altitude = float(final_pos[2])
    altitude_loss = initial_altitude - min_altitude

    metrics = {
        "target_distance": float(distance_cm),
        "target_heading": float(heading_deg),
        "success": bool(success),
        "final_position": [round(float(v), 4) for v in final_pos],
        "min_target_distance": round(min_target_dist, 4),
        "initial_altitude": round(initial_altitude, 4),
        "min_altitude": round(min_altitude, 4),
        "final_altitude": round(final_altitude, 4),
        "altitude_loss": round(altitude_loss, 4),
        "flight_duration": round(step_count * dt, 4),
        "max_heading_deviation": round(max_heading_dev, 2),
        "touchdown": bool(touchdown),
        "collision": bool(collision),
        "timeout": bool(timeout),
    }

    return metrics, frames


def main():
    print("=" * 70)
    print(" MILESTONE M4A: EMPIRICAL FLIGHT ENVELOPE BENCHMARK")
    print("=" * 70)

    distances = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
    headings = [0, 10, 20, 30, 35, 40, 45]
    n_reps = 3

    out_dir = os.path.join(WORKSPACE_ROOT, "results", "m4a_envelope")
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(WORKSPACE_ROOT, "videos")
    os.makedirs(video_dir, exist_ok=True)

    print(f"Loading FlyBody imitation environment and neural policy controller...")
    env = fly_envs.flight_imitation(ref_path=None, terminal_com_dist=float('inf'))
    controller = FlightPolicyController()

    all_records: List[Dict[str, Any]] = []
    demo_video_frames: List[np.ndarray] = []

    # Selected cases for video recording
    video_cases = [
        (0.50, 0),    # Short straight (safe)
        (1.00, 20),   # Medium turn (envelope boundary)
        (3.00, 0),    # Long straight (altitude decay)
        (3.00, 45),   # Long steep turn (severe failure)
    ]

    total_runs = len(distances) * len(headings) * n_reps
    run_idx = 0
    t_start = time.time()

    print(f"Executing {total_runs} simulation episodes across {len(distances)} distances and {len(headings)} headings...\n")

    for dist in distances:
        for heading in headings:
            for rep in range(n_reps):
                run_idx += 1
                should_record = (rep == 0) and ((dist, heading) in video_cases)
                metrics, frames = run_single_episode(
                    env, controller, dist, heading, record_frames=should_record
                )
                metrics["rep"] = rep
                all_records.append(metrics)

                if should_record and frames:
                    demo_video_frames.extend(frames)

            # Log summary for condition
            cond_records = [r for r in all_records if r["target_distance"] == dist and r["target_heading"] == heading]
            succ_rate = np.mean([1 if r["success"] else 0 for r in cond_records]) * 100
            mean_alt_loss = np.mean([r["altitude_loss"] for r in cond_records])
            td_rate = np.mean([1 if r["touchdown"] else 0 for r in cond_records]) * 100
            print(f"[{run_idx:03d}/{total_runs}] Dist: {dist:4.2f} cm | Head: {heading:2d}° -> Success: {succ_rate:5.1f}% | Alt Loss: {mean_alt_loss:.3f} cm | Touchdown: {td_rate:4.0f}%")

    elapsed = time.time() - t_start
    print(f"\nAll {total_runs} episodes completed in {elapsed:.2f} s ({elapsed/total_runs*1000:.1f} ms/ep).")

    # 1. Save JSON
    json_path = os.path.join(out_dir, "flight_envelope_results.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved complete JSON results: {json_path}")

    # 2. Save CSV
    csv_path = os.path.join(out_dir, "flight_envelope_results.csv")
    fieldnames = list(all_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved complete CSV results: {csv_path}")

    # 3. Generate Analysis Figures
    print("\nGenerating empirical flight envelope plots...")
    generate_envelope_plots(all_records, distances, headings, out_dir)

    # 4. Save Representative Animation
    if demo_video_frames:
        try:
            vid_webp = os.path.join(video_dir, "m4a_envelope_tests.webp")
            pil_frames = [Image.fromarray(f) for f in demo_video_frames[::2]]
            pil_frames[0].save(
                vid_webp,
                save_all=True,
                append_images=pil_frames[1:],
                duration=66,
                loop=0
            )
            print(f"Saved WebP test animation: {vid_webp}")
        except Exception as e:
            print(f"WebP save error: {e}")


def generate_envelope_plots(
    records: List[Dict[str, Any]],
    distances: List[float],
    headings: List[int],
    out_dir: str
):
    """Creates publication-quality 2D heatmaps and altitude degradation curves."""
    if not HAS_MATPLOTLIB:
        print("[WARNING] Matplotlib not available; skipping plot generation.")
        return
    success_matrix = np.zeros((len(distances), len(headings)))
    alt_loss_matrix = np.zeros((len(distances), len(headings)))
    touchdown_matrix = np.zeros((len(distances), len(headings)))

    for i, d in enumerate(distances):
        for j, h in enumerate(headings):
            matches = [r for r in records if r["target_distance"] == d and r["target_heading"] == h]
            success_matrix[i, j] = np.mean([1 if m["success"] else 0 for m in matches]) * 100.0
            alt_loss_matrix[i, j] = np.mean([m["altitude_loss"] for m in matches])
            touchdown_matrix[i, j] = np.mean([1 if m["touchdown"] else 0 for m in matches]) * 100.0

    # Figure 1: Heatmap Grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. Success Rate Heatmap
    im1 = axes[0].imshow(success_matrix, cmap="RdYlGn", origin="lower", aspect="auto", vmin=0, vmax=100)
    axes[0].set_title("Navigation Success Rate (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Target Heading Change (deg)", fontsize=10)
    axes[0].set_ylabel("Target Distance (cm)", fontsize=10)
    axes[0].set_xticks(range(len(headings)))
    axes[0].set_xticklabels([f"{h}°" for h in headings])
    axes[0].set_yticks(range(len(distances)))
    axes[0].set_yticklabels([f"{d:.2f}" for d in distances])
    plt.colorbar(im1, ax=axes[0], label="Success Rate (%)")

    # Annotate values
    for i in range(len(distances)):
        for j in range(len(headings)):
            val = success_matrix[i, j]
            color = "black" if 30 <= val <= 70 else "white"
            axes[0].text(j, i, f"{int(val)}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    # 2. Altitude Loss Heatmap
    im2 = axes[1].imshow(alt_loss_matrix, cmap="magma_r", origin="lower", aspect="auto")
    axes[1].set_title("Mean Altitude Loss (cm)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Target Heading Change (deg)", fontsize=10)
    axes[1].set_xticks(range(len(headings)))
    axes[1].set_xticklabels([f"{h}°" for h in headings])
    axes[1].set_yticks(range(len(distances)))
    axes[1].set_yticklabels([f"{d:.2f}" for d in distances])
    plt.colorbar(im2, ax=axes[1], label="Altitude Loss (cm)")

    for i in range(len(distances)):
        for j in range(len(headings)):
            val = alt_loss_matrix[i, j]
            color = "white" if val > 0.45 else "black"
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    # 3. Touchdown Rate Heatmap
    im3 = axes[2].imshow(touchdown_matrix, cmap="Reds", origin="lower", aspect="auto", vmin=0, vmax=100)
    axes[2].set_title("Touchdown / Floor Contact Rate (%)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Target Heading Change (deg)", fontsize=10)
    axes[2].set_xticks(range(len(headings)))
    axes[2].set_xticklabels([f"{h}°" for h in headings])
    axes[2].set_yticks(range(len(distances)))
    axes[2].set_yticklabels([f"{d:.2f}" for d in distances])
    plt.colorbar(im3, ax=axes[2], label="Touchdown Rate (%)")

    for i in range(len(distances)):
        for j in range(len(headings)):
            val = touchdown_matrix[i, j]
            color = "white" if val > 50 else "black"
            axes[2].text(j, i, f"{int(val)}%", ha="center", va="center", color=color, fontsize=8)

    plt.suptitle("FlyBody Flight Controller Operating Envelope (Milestone M4A)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path1 = os.path.join(out_dir, "m4a_flight_envelope_heatmaps.png")
    plt.savefig(plot_path1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved envelope heatmap plot: {plot_path1}")

    # Figure 2: Altitude Decay Curves vs Distance for Different Headings
    plt.figure(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(headings)))
    for j, h in enumerate(headings):
        losses = alt_loss_matrix[:, j]
        plt.plot(distances, losses, marker="o", linewidth=2, label=f"Heading {h}°", color=colors[j])

    plt.axhline(y=1.037 - 0.20, color="red", linestyle="--", linewidth=1.5, label="Touchdown Limit (Δz = 0.837 cm)")
    plt.title("Altitude Loss as a Function of Target Distance", fontsize=12, fontweight="bold")
    plt.xlabel("Target Distance (cm)", fontsize=11)
    plt.ylabel("Altitude Loss (cm)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plot_path2 = os.path.join(out_dir, "m4a_altitude_decay_curves.png")
    plt.savefig(plot_path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved altitude decay curve plot: {plot_path2}")


if __name__ == "__main__":
    main()
