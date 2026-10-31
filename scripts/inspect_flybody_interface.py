"""
FlyBody Interface Inspection & Characterization Tool (Milestone 2).
Thoroughly inspects, measures, and visualizes:
1. Exact observation spaces, dimensions, units, frames, and statistics
2. Exact 12-channel flight action space, bounds, actuators, and semantics
3. Control loop timings, frequencies, and ratios
4. Action perturbation dynamics (differential wing roll, pitch, frequency bias, body joints)
5. Exports machine-readable JSON metrics and plots to results/milestone_02/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flybody.fly_envs import flight_imitation, vision_guided_flight
from src.data.flight_dataset import quat_to_euler


def inspect_environment():
    print("=" * 65)
    print(" FLYBODY INTERFACE INSPECTION (MILESTONE 2)")
    print("=" * 65)

    # 1. Instantiate flight_imitation
    env = flight_imitation(terminal_com_dist=float('inf'))
    time_step = env.reset()
    physics = env.physics
    walker = env.task._walker
    task = env.task

    obs_spec = env.observation_spec()
    act_spec = env.action_spec()

    # Timing
    ctrl_dt = env.control_timestep()
    sim_dt = physics.model.opt.timestep
    ctrl_freq = 1.0 / ctrl_dt
    sim_freq = 1.0 / sim_dt
    substeps = int(round(ctrl_dt / sim_dt))

    print(f"\nEnvironment: flight_imitation")
    print(f"Control frequency: {ctrl_freq:.0f} Hz (dt = {ctrl_dt:.6f} s)")
    print(f"Physics frequency: {sim_freq:.0f} Hz (dt = {sim_dt:.6f} s)")
    print(f"Physics-to-control ratio: {substeps} physics substeps per control step")

    # 2. Inspect Observation Space
    obs_dict = time_step.observation
    total_obs_dim = sum(int(np.prod(v.shape)) for v in obs_dict.values())
    print(f"\nObservation Space:")
    print(f"  Total flattened dimension: {total_obs_dim}")
    print(f"  Number of named observation fields: {len(obs_dict)}")

    obs_stats = {}
    current_idx = 0
    obs_table = []

    observable_joint_names = [j.name for j in walker.observable_joints]

    for k, v in obs_dict.items():
        shape = list(v.shape)
        size = int(np.prod(v.shape))
        dtype_str = str(v.dtype)
        val_min = float(np.min(v)) if size > 0 else 0.0
        val_max = float(np.max(v)) if size > 0 else 0.0
        val_mean = float(np.mean(v)) if size > 0 else 0.0

        obs_stats[k] = {
            "start_index": current_idx,
            "end_index": current_idx + size,
            "size": size,
            "shape": shape,
            "dtype": dtype_str,
            "sample_min": val_min,
            "sample_max": val_max,
            "sample_mean": val_mean,
        }

        obs_table.append({
            "field": k,
            "indices": f"{current_idx}..{current_idx + size - 1}" if size > 0 else "N/A",
            "size": size,
            "shape": shape,
            "dtype": dtype_str,
        })
        current_idx += size

    for item in obs_table:
        print(f"  - {item['field']:<30} | Dims: {item['size']:<3} | Shape: {str(item['shape']):<10} | Indices: {item['indices']}")

    # 3. Inspect Action Space
    act_shape = act_spec.shape
    act_min = act_spec.minimum
    act_max = act_spec.maximum
    act_names = act_spec.name.split('\t') if hasattr(act_spec, 'name') and act_spec.name else [f"action_{i}" for i in range(act_shape[0])]

    print(f"\nAction Space:")
    print(f"  Shape: {act_shape}")
    print(f"  Dtype: {act_spec.dtype}")
    print(f"  Number of channels: {act_shape[0]}")

    action_stats = {}
    action_table = []

    act_descriptions = [
        ("head_abduct", "Head roll / lateral abduction angle", "radians", "Head roll stabilization"),
        ("head_twist", "Head yaw / twist angle", "radians", "Head gaze / yaw tracking"),
        ("head", "Head pitch / flexion-extension angle", "radians", "Head pitch stabilization"),
        ("wing_yaw_left", "Left wing stroke angle offset (bias)", "radians", "Modulates stroke forward/backward amplitude"),
        ("wing_roll_left", "Left wing deviation / elevation offset", "radians", "Modulates stroke plane elevation / bank"),
        ("wing_pitch_left", "Left wing feathering / angle-of-attack offset", "radians", "Modulates wing lift and drag coefficients"),
        ("wing_yaw_right", "Right wing stroke angle offset (bias)", "radians", "Modulates stroke forward/backward amplitude"),
        ("wing_roll_right", "Right wing deviation / elevation offset", "radians", "Modulates stroke plane elevation / bank"),
        ("wing_pitch_right", "Right wing feathering / angle-of-attack offset", "radians", "Modulates wing lift and drag coefficients"),
        ("abdomen_abduct", "Abdominal lateral steering / yaw angle", "radians", "Trims body yaw torque in flight"),
        ("abdomen", "Abdominal flexion / pitch angle", "radians", "Trims body pitch attitude and CoM location"),
        ("wbpg_frequency_bias", "Normalized wingbeat flapping frequency bias", "[-1, 1]", "Scales base flapping freq +/- 10% (lift gain)"),
    ]

    for i in range(act_shape[0]):
        name = act_names[i] if i < len(act_names) else f"act_{i}"
        desc = act_descriptions[i][1] if i < len(act_descriptions) else "Control parameter"
        units = act_descriptions[i][2] if i < len(act_descriptions) else "arb"
        effect = act_descriptions[i][3] if i < len(act_descriptions) else "Kinematic modulation"

        action_stats[f"channel_{i}"] = {
            "name": name,
            "min": float(act_min[i]),
            "max": float(act_max[i]),
            "units": units,
            "description": desc,
            "effect": effect,
        }

        print(f"  [{i:02d}] {name:<22} | Range: [{act_min[i]:>6.2f}, {act_max[i]:>6.2f}] {units:<8} | {effect}")

    # Output directory
    out_dir = os.path.join(WORKSPACE_ROOT, "results", "milestone_02")
    plots_dir = os.path.join(out_dir, "plots")
    perturb_dir = os.path.join(out_dir, "action_perturbation")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(perturb_dir, exist_ok=True)

    # Save JSON files
    with open(os.path.join(out_dir, "observation_stats.json"), "w") as f:
        json.dump(obs_stats, f, indent=2)
    with open(os.path.join(out_dir, "action_stats.json"), "w") as f:
        json.dump(action_stats, f, indent=2)

    print(f"\nSaved stats JSONs to {out_dir}")

    # 4. Generate Visualization Plots
    plot_observation_action_breakdown(obs_stats, action_stats, plots_dir)

    # 5. Run Controlled Action Perturbation Experiments
    run_perturbation_experiments(env, perturb_dir)

    print("=" * 65)
    print(" MILESTONE 2 INSPECTION COMPLETE")
    print("=" * 65)


def plot_observation_action_breakdown(obs_stats, action_stats, plots_dir):
    """Generates visual breakdown charts of observation and action dimensions."""
    # 1. Observation distribution pie / bar chart
    labels = []
    sizes = []
    for k, v in obs_stats.items():
        clean_name = k.replace("walker/", "")
        labels.append(f"{clean_name} ({v['size']})")
        sizes.append(v['size'])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, sizes, color=colors)
    ax.set_xlabel("Number of Continuous Observable Dimensions", fontsize=11)
    ax.set_title("FlyBody Flight Imitation Observation Space Breakdown (Total = 104 dims)", fontsize=13, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "observation_dimensions_breakdown.png"))
    plt.close()

    # 2. Action range chart
    act_names = [v["name"] for v in action_stats.values()]
    mins = [v["min"] for v in action_stats.values()]
    maxs = [v["max"] for v in action_stats.values()]
    indices = np.arange(len(act_names))

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for i in range(len(act_names)):
        ax.plot([mins[i], maxs[i]], [i, i], color='#1f77b4', lw=4, solid_capstyle='round')
        ax.scatter([mins[i], maxs[i]], [i, i], color='#d62728', s=40, zorder=3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(indices)
    ax.set_yticklabels(act_names, fontsize=10)
    ax.set_xlabel("Permissible Action Range [min, max]", fontsize=11)
    ax.set_title("FlyBody 12-Channel Action Bounds & Control Extents", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "action_bounds_ranges.png"))
    plt.close()
    print(f"Generated interface plots in {plots_dir}")


def run_perturbation_experiments(env, perturb_dir):
    """
    Executes controlled action perturbation rollouts to quantify
    the exact physical controllability of the fly.
    """
    print(f"\nRunning Action Perturbation Controllability Study...")

    experiments = [
        ("baseline_zero_action", np.zeros(12, dtype=np.float32), "Baseline (Zero Kinematic Bias)"),
        ("differential_wing_roll_bank", np.array([0, 0, 0, 0, 0.5, 0, 0, -0.5, 0, 0, 0, 0], dtype=np.float32), "Differential Wing Roll (+0.5 L, -0.5 R)"),
        ("symmetric_wing_pitch_thrust", np.array([0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0], dtype=np.float32), "Symmetric Wing Pitch (+0.5 AoA)"),
        ("frequency_boost_climb", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32), "Max Wing Flapping Frequency (+100% Bias)"),
        ("abdominal_steering_yaw", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0], dtype=np.float32), "Abdominal Lateral Steering (+0.5 rad)"),
    ]

    results = {}
    time_horizon_steps = 150  # 150 steps @ 0.0002 s = 0.03 seconds

    for exp_id, test_action, exp_name in experiments:
        env.reset()
        physics = env.physics

        pos_traj = []
        vel_traj = []
        euler_traj = []
        time_traj = []

        for step_i in range(time_horizon_steps):
            t = step_i * env.control_timestep()
            env.step(test_action)

            qpos = physics.data.qpos[:7]
            qvel = physics.data.qvel[:6]
            pos = qpos[:3]
            quat = qpos[3:7]
            euler = quat_to_euler(quat)

            pos_traj.append(np.copy(pos))
            vel_traj.append(np.copy(qvel[:3]))
            euler_traj.append(np.copy(euler))
            time_traj.append(t)

        pos_arr = np.array(pos_traj)
        vel_arr = np.array(vel_traj)
        euler_arr = np.array(euler_traj)
        time_arr = np.array(time_traj)

        delta_pos = pos_arr[-1] - pos_arr[0]
        final_speed = np.linalg.norm(vel_arr[-1])
        final_roll = euler_arr[-1, 0]
        final_pitch = euler_arr[-1, 1]
        final_yaw = euler_arr[-1, 2]

        results[exp_id] = {
            "name": exp_name,
            "delta_pos_cm": [float(x) for x in delta_pos],
            "net_displacement_cm": float(np.linalg.norm(delta_pos)),
            "final_speed_cms": float(final_speed),
            "final_euler_deg": [float(np.rad2deg(x)) for x in [final_roll, final_pitch, final_yaw]],
        }

        # Save trajectory plot
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, dpi=150)
        axes[0].plot(time_arr * 1000, pos_arr[:, 0], label="X (Forward)", color='#1f77b4')
        axes[0].plot(time_arr * 1000, pos_arr[:, 1], label="Y (Lateral)", color='#ff7f0e')
        axes[0].plot(time_arr * 1000, pos_arr[:, 2], label="Z (Altitude)", color='#2ca02c')
        axes[0].set_ylabel("CoM Position (cm)")
        axes[0].set_title(f"Perturbation Dynamics: {exp_name}", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        axes[1].plot(time_arr * 1000, vel_arr[:, 0], label="Vx", color='#1f77b4')
        axes[1].plot(time_arr * 1000, vel_arr[:, 1], label="Vy", color='#ff7f0e')
        axes[1].plot(time_arr * 1000, vel_arr[:, 2], label="Vz", color='#2ca02c')
        axes[1].set_ylabel("Linear Velocity (cm/s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left")

        axes[2].plot(time_arr * 1000, np.rad2deg(euler_arr[:, 0]), label="Roll (Bank)", color='#d62728')
        axes[2].plot(time_arr * 1000, np.rad2deg(euler_arr[:, 1]), label="Pitch", color='#9467bd')
        axes[2].plot(time_arr * 1000, np.rad2deg(euler_arr[:, 2]), label="Yaw", color='#8c564b')
        axes[2].set_ylabel("Attitude (degrees)")
        axes[2].set_xlabel("Time (milliseconds)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(perturb_dir, f"{exp_id}.png"))
        plt.close()

    with open(os.path.join(perturb_dir, "perturbation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved perturbation trajectories and summary JSON to {perturb_dir}")


if __name__ == "__main__":
    inspect_environment()
