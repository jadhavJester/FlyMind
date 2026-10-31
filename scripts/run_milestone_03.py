"""
Milestone 3 Execution Script: Drosophila Arithmetic Selector v0.1.

Executes the complete embodied arithmetic task experiment:
1. Layer 1 (Task): Generates '2 + 3 = ?' with candidates [5, 7].
2. Layer 2 (Decision): Deterministic arithmetic solving (2 + 3 = 5).
3. Layer 3 (Navigation): Generates target-directed reference trajectory.
4. Layer 4 (FlyBody): Executes physical flight in MuJoCo physics simulation.

Evaluates 3 Baselines:
- Baseline A: Oracle Target (directly commands correct target 'target_left')
- Baseline B: Wrong Target (explicitly commands incorrect target 'target_right')
- Baseline C: Arithmetic Solver (end-to-end task -> decision -> flight -> contact)

Outputs:
- Structured episode logs: results/milestone_03/episode_logs.json
- Quantitative summary: results/milestone_03/summary.json
- Comparative trajectory plot: results/milestone_03/trajectory_comparison.png
- High-fidelity telemetry video: videos/milestone_03_arithmetic.mp4 and .webp
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imageio

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.tasks.arithmetic import ArithmeticTaskGenerator
from src.agents.oracle_selector import (
    OracleSelectorAgent,
    WrongTargetSelectorAgent,
    ArithmeticDecisionAgent,
    FlightPolicyController
)
from src.environments.interactive_flight import InteractiveArithmeticEnv


def draw_hud(
    track_frame: np.ndarray,
    top_frame: np.ndarray,
    task_info: dict,
    mode_name: str,
    selected_target_id: str,
    step: int,
    max_steps: int,
    sim_time_ms: float,
    fly_pos: np.ndarray,
    dist_left: float,
    dist_right: float,
    current_reward: float,
    cumulative_reward: float,
    status_text: str,
    is_terminal: bool = False
) -> np.ndarray:
    """
    Composites dual-camera frames (track1 + top_camera) with a telemetry HUD.
    Total resolution: 960 x 540.
    """
    canvas_w = 960
    canvas_h = 540
    header_h = 100
    footer_h = 80
    view_h = canvas_h - header_h - footer_h  # 360 px
    view_w = canvas_w // 2                  # 480 px

    # Resize camera frames
    img_track = Image.fromarray(track_frame).resize((view_w, view_h))
    img_top = Image.fromarray(top_frame).resize((view_w, view_h))

    # Base canvas (dark theme)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 20, 28))
    draw = ImageDraw.Draw(canvas)

    # Paste camera views
    canvas.paste(img_track, (0, header_h))
    canvas.paste(img_top, (view_w, header_h))

    # Camera Labels
    draw.rectangle([10, header_h + 10, 180, header_h + 32], fill=(0, 0, 0, 180))
    draw.text((16, header_h + 14), "PERSPECTIVE: 3D TRACK", fill=(200, 220, 255))

    draw.rectangle([view_w + 10, header_h + 10, view_w + 195, header_h + 32], fill=(0, 0, 0, 180))
    draw.text((view_w + 16, header_h + 14), "ARENA: TOP-DOWN (X-Y)", fill=(200, 220, 255))

    # Top-down camera target markers overlay
    # Top view camera covers roughly X in [-2, 8] and Y in [-5, 5]
    # Draw simple visual target circles on the top view
    top_cx = view_w + view_w // 2
    top_cy = header_h + view_h // 2
    # Map fly and targets roughly
    scale_y = 35.0  # px per cm
    scale_x = 35.0

    target_left_x = int(top_cx + 0.60 * scale_x)
    target_left_y = int(top_cy - 0.25 * scale_y)  # +Y is up on screen
    target_right_x = int(top_cx + 0.60 * scale_x)
    target_right_y = int(top_cy + 0.25 * scale_y)  # -Y is down on screen

    # Left Target [5] (Green circle)
    draw.ellipse([target_left_x - 12, target_left_y - 12, target_left_x + 12, target_left_y + 12],
                 outline=(40, 220, 80), width=2, fill=(20, 80, 40))
    draw.text((target_left_x - 4, target_left_y - 6), "5", fill=(255, 255, 255))

    # Right Target [7] (Orange/Red circle)
    draw.ellipse([target_right_x - 12, target_right_y - 12, target_right_x + 12, target_right_y + 12],
                 outline=(240, 100, 50), width=2, fill=(80, 30, 20))
    draw.text((target_right_x - 4, target_right_y - 6), "7", fill=(255, 255, 255))

    # Fly position marker on top view
    fly_mx = int(top_cx + fly_pos[0] * scale_x)
    fly_my = int(top_cy - fly_pos[1] * scale_y)
    draw.ellipse([fly_mx - 5, fly_my - 5, fly_mx + 5, fly_my + 5], fill=(255, 230, 50))

    # Divider line between cameras
    draw.line([(view_w, header_h), (view_w, header_h + view_h)], fill=(40, 50, 70), width=2)

    # HEADER BAR
    draw.rectangle([0, 0, canvas_w, header_h], fill=(18, 24, 38))
    draw.line([(0, header_h), (canvas_w, header_h)], fill=(40, 60, 90), width=2)

    # Title & Task
    draw.text((20, 12), "FLYBODY MILESTONE 3: DROSOPHILA ARITHMETIC SELECTOR v0.1", fill=(140, 180, 240))
    
    # Task Prompt Box
    draw.rectangle([20, 36, 260, 88], fill=(28, 38, 58), outline=(60, 90, 140))
    draw.text((32, 42), "TASK PROMPT:", fill=(160, 175, 200))
    draw.text((32, 58), task_info["prompt_str"], fill=(255, 255, 255))

    # Candidate Targets Box
    left_val = task_info["candidate_target_mapping"]["target_left"]
    right_val = task_info["candidate_target_mapping"]["target_right"]
    is_left_sel = (selected_target_id == "target_left")

    # Left Target HUD Card
    left_bg = (30, 70, 45) if is_left_sel else (24, 32, 48)
    left_border = (50, 220, 90) if is_left_sel else (60, 80, 110)
    draw.rectangle([280, 36, 430, 88], fill=left_bg, outline=left_border, width=2 if is_left_sel else 1)
    draw.text((290, 42), f"TARGET LEFT: [{left_val}]", fill=(100, 240, 140) if is_left_sel else (180, 190, 210))
    draw.text((290, 64), f"Dist: {dist_left:.2f} cm", fill=(220, 240, 230))

    # Right Target HUD Card
    is_right_sel = (selected_target_id == "target_right")
    right_bg = (70, 35, 30) if is_right_sel else (24, 32, 48)
    right_border = (240, 90, 70) if is_right_sel else (60, 80, 110)
    draw.rectangle([450, 36, 600, 88], fill=right_bg, outline=right_border, width=2 if is_right_sel else 1)
    draw.text((460, 42), f"TARGET RIGHT: [{right_val}]", fill=(255, 140, 120) if is_right_sel else (180, 190, 210))
    draw.text((460, 64), f"Dist: {dist_right:.2f} cm", fill=(240, 220, 220))

    # Mode & Decision Box
    draw.rectangle([620, 36, 940, 88], fill=(28, 38, 58), outline=(60, 90, 140))
    draw.text((632, 42), f"MODE: {mode_name}", fill=(255, 210, 80))
    target_label = f"[{left_val}]" if is_left_sel else f"[{right_val}]"
    draw.text((632, 62), f"Selected: {selected_target_id} {target_label}", fill=(200, 240, 255))

    # FOOTER BAR
    draw.rectangle([0, canvas_h - footer_h, canvas_w, canvas_h], fill=(18, 24, 38))
    draw.line([(0, canvas_h - footer_h), (canvas_w, canvas_h - footer_h)], fill=(40, 60, 90), width=2)

    # Telemetry data
    draw.text((20, canvas_h - footer_h + 12),
              f"STEP: {step:3d} / {max_steps}   |   SIM TIME: {sim_time_ms:5.1f} ms   |   FLY POS: [{fly_pos[0]:.2f}, {fly_pos[1]:.2f}, {fly_pos[2]:.2f}] cm",
              fill=(180, 200, 230))

    # Status / Reward banner
    status_color = (80, 240, 120) if "SUCCESS" in status_text or "CORRECT" in status_text else \
                   (255, 90, 80) if "WRONG" in status_text or "COLLISION" in status_text else (240, 210, 70)
    draw.text((20, canvas_h - footer_h + 40),
              f"STATUS: {status_text}   |   STEP REWARD: {current_reward:+6.2f}   |   CUMULATIVE: {cumulative_reward:+6.2f}",
              fill=status_color)

    return np.array(canvas)


def run_episode(
    env: InteractiveArithmeticEnv,
    controller: FlightPolicyController,
    task,
    agent,
    mode_name: str,
    record_frames: bool = True
) -> dict:
    """
    Executes a single complete experimental episode.
    """
    # 1. High-level agent selects target
    decision = agent.select_target(task)
    selected_target_id = decision["selected_target_id"]

    # 2. Reset environment with task and selected target
    obs = env.reset(task=task, selected_target_id=selected_target_id)
    initial_pos = env.get_fly_position().tolist()

    frames = []
    trajectory = [env.get_fly_position().copy()]
    rewards = []
    cumulative_reward = 0.0
    terminated = False
    termination_reason = "RUNNING"
    info = {}

    status_text = "NAVIGATING TO SELECTED TARGET"

    for step in range(env.max_steps):
        # Layer 3 & 4: Compute action via policy controller
        action = controller.get_action(obs)

        # Step live MuJoCo physics
        obs, reward, terminated, info = env.step(action)
        fly_pos = env.get_fly_position()
        trajectory.append(fly_pos.copy())
        rewards.append(reward)
        cumulative_reward += reward

        dist_left = info["dist_left"]
        dist_right = info["dist_right"]

        if terminated:
            termination_reason = info["termination_reason"]
            status_text = f"TERMINATED: {termination_reason}"
            if info["success"]:
                status_text = f"SUCCESS: REACHED CORRECT TARGET [{task.correct_answer}]"
            elif termination_reason == "WRONG_TARGET_CONTACT":
                status_text = "FAILED: CONTACTED WRONG DISTRACTOR TARGET"
        else:
            status_text = f"FLYING TOWARDS {selected_target_id.upper()}"

        # Render offscreen camera frames every 2 control steps (2500 Hz video capture)
        if record_frames and (step % 2 == 0 or terminated):
            track_frame = env.physics.render(camera_id='walker/track1', height=360, width=480)
            top_frame = env.physics.render(camera_id='top_camera', height=360, width=480)
            sim_time_ms = step * env.dt * 1000.0

            hud_frame = draw_hud(
                track_frame=track_frame,
                top_frame=top_frame,
                task_info=task.to_dict(),
                mode_name=mode_name,
                selected_target_id=selected_target_id,
                step=step,
                max_steps=env.max_steps,
                sim_time_ms=sim_time_ms,
                fly_pos=fly_pos,
                dist_left=dist_left,
                dist_right=dist_right,
                current_reward=reward,
                cumulative_reward=cumulative_reward,
                status_text=status_text,
                is_terminal=terminated
            )
            frames.append(hud_frame)

        if terminated:
            break

    traj_arr = np.array(trajectory)
    dist_travelled = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1)))

    episode_data = {
        "mode_name": mode_name,
        "task_id": task.task_id,
        "prompt": task.prompt_str,
        "candidates": task.candidates,
        "correct_answer": task.correct_answer,
        "selected_target_id": selected_target_id,
        "selected_value": task.candidate_target_mapping[selected_target_id],
        "initial_pos": initial_pos,
        "final_pos": traj_arr[-1].tolist(),
        "steps": len(rewards),
        "duration_seconds": len(rewards) * env.dt,
        "distance_travelled_cm": dist_travelled,
        "min_dist_correct_cm": float(info.get("min_dist_correct", 0.0)),
        "min_dist_wrong_cm": float(info.get("min_dist_wrong", 0.0)),
        "success": bool(info.get("success", False)),
        "cumulative_reward": float(cumulative_reward),
        "termination_reason": termination_reason,
        "trajectory": traj_arr.tolist()
    }

    return episode_data, frames


def main():
    print("=" * 70)
    print(" FlyBody Milestone 3: Drosophila Arithmetic Selector v0.1")
    print("=" * 70)

    results_dir = os.path.join(WORKSPACE_ROOT, "results", "milestone_03")
    videos_dir = os.path.join(WORKSPACE_ROOT, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # 1. Initialize Task Generator
    task_gen = ArithmeticTaskGenerator()
    task = task_gen.generate_default_v01_task()
    print(f"Task Generated: {task.prompt_str}")
    print(f"Candidate Answers: Left = [{task.candidate_target_mapping['target_left']}], "
          f"Right = [{task.candidate_target_mapping['target_right']}]")
    print(f"Correct Answer: {task.correct_answer} (Target: {task.correct_target_id})")

    # 2. Initialize Environment and Flight Controller
    print("Initializing Interactive FlyBody Environment...")
    env = InteractiveArithmeticEnv(
        target_distance_x=0.60,
        target_offset_y=0.25,
        target_height_z=0.55,
        contact_radius=0.15,
        max_steps=220
    )
    print("Loading Flight Policy Controller...")
    controller = FlightPolicyController()

    # 3. Instantiate Baseline Agents
    oracle_agent = OracleSelectorAgent()
    wrong_agent = WrongTargetSelectorAgent()
    solver_agent = ArithmeticDecisionAgent()

    baselines = [
        ("Baseline A (Oracle)", oracle_agent),
        ("Baseline B (Wrong Target)", wrong_agent),
        ("Baseline C (Arithmetic Solver)", solver_agent),
    ]

    all_logs = []
    all_frames = []
    trajectories = {}

    for mode_name, agent in baselines:
        print("-" * 70)
        print(f"Running Experiment: {mode_name} ...")
        ep_data, frames = run_episode(
            env=env,
            controller=controller,
            task=task,
            agent=agent,
            mode_name=mode_name,
            record_frames=True
        )

        all_logs.append(ep_data)
        trajectories[mode_name] = np.array(ep_data["trajectory"])
        all_frames.extend(frames)

        # Save snapshot frame of baseline
        if len(frames) > 0:
            frame_filename = os.path.join(results_dir, f"{mode_name[:10].lower().replace(' ', '_')}_final.png")
            Image.fromarray(frames[-1]).save(frame_filename)

        print(f"  Result: Success={ep_data['success']}, Steps={ep_data['steps']}, "
              f"Reward={ep_data['cumulative_reward']:.2f}, Termination={ep_data['termination_reason']}")
        print(f"  Min Dist to Correct [{task.correct_answer}]: {ep_data['min_dist_correct_cm']:.3f} cm")
        print(f"  Min Dist to Wrong   [{task.candidates[1]}]: {ep_data['min_dist_wrong_cm']:.3f} cm")

    # 4. Save JSON Logs
    log_path = os.path.join(results_dir, "episode_logs.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\nSaved structured episode logs to: {log_path}")

    # Summary JSON
    summary_path = os.path.join(results_dir, "summary.json")
    summary = {
        "milestone": "Milestone 3 — Drosophila Arithmetic Selector v0.1",
        "task": task.to_dict(),
        "baselines": {
            log["mode_name"]: {
                "selected_target": log["selected_target_id"],
                "selected_value": log["selected_value"],
                "success": log["success"],
                "cumulative_reward": log["cumulative_reward"],
                "steps": log["steps"],
                "min_dist_correct_cm": log["min_dist_correct_cm"],
                "min_dist_wrong_cm": log["min_dist_wrong_cm"],
                "termination_reason": log["termination_reason"]
            }
            for log in all_logs
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved quantitative summary to: {summary_path}")

    # 5. Generate Comparative Trajectory Plot
    plot_path = os.path.join(results_dir, "trajectory_comparison.png")
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Top-down (X-Y)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("3D Arena Flight Path (Top-Down X-Y)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (Forward, cm)")
    ax1.set_ylabel("Y (Lateral, cm)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot targets
    t_left = env.target_left_pos
    t_right = env.target_right_pos
    circle_left = plt.Circle((t_left[0], t_left[1]), env.contact_radius, color='green', alpha=0.3, label="[5] Target Left (Correct)")
    circle_right = plt.Circle((t_right[0], t_right[1]), env.contact_radius, color='red', alpha=0.3, label="[7] Target Right (Wrong)")
    ax1.add_patch(circle_left)
    ax1.add_patch(circle_right)
    ax1.plot(t_left[0], t_left[1], 'go', markersize=8)
    ax1.plot(t_right[0], t_right[1], 'ro', markersize=8)

    colors = {
        "Baseline A (Oracle)": "blue",
        "Baseline B (Wrong Target)": "crimson",
        "Baseline C (Arithmetic Solver)": "forestgreen"
    }

    for name, traj in trajectories.items():
        ax1.plot(traj[:, 0], traj[:, 1], label=name, color=colors.get(name, "black"), linewidth=2.0)
        ax1.plot(traj[0, 0], traj[0, 1], 'ks', markersize=5)  # Start
        ax1.plot(traj[-1, 0], traj[-1, 1], 'x', color=colors.get(name, "black"), markersize=8)  # End

    ax1.legend(loc="upper left", fontsize=9)

    # Subplot 2: Altitude profile (X-Z)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Altitude Profile (X-Z)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("X (Forward, cm)")
    ax2.set_ylabel("Z (Altitude, cm)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2.axhline(0.20, color='brown', linestyle=':', label='Ground Crash Floor (0.20 cm)')
    ax2.plot(t_left[0], t_left[2], 'go', markersize=8, label='Target Left Z')
    ax2.plot(t_right[0], t_right[2], 'ro', markersize=8, label='Target Right Z')

    for name, traj in trajectories.items():
        ax2.plot(traj[:, 0], traj[:, 2], label=name, color=colors.get(name, "black"), linewidth=2.0)

    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparative trajectory plot to: {plot_path}")

    # 6. Render Video
    if len(all_frames) > 0:
        mp4_path = os.path.join(videos_dir, "milestone_03_arithmetic.mp4")
        webp_path = os.path.join(videos_dir, "milestone_03_arithmetic.webp")
        print(f"Exporting experiment video ({len(all_frames)} frames @ 30 fps)...")

        imageio.mimsave(mp4_path, all_frames, fps=30, quality=8)
        print(f"Saved MP4 video to: {mp4_path}")

        webp_frames = all_frames[::2]
        imageio.mimsave(webp_path, webp_frames, fps=15)
        print(f"Saved WebP animation to: {webp_path}")

    print("\n[NOTE] By default, this script runs in headless offscreen mode and exports video.")
    print("To open the interactive LIVE 3D simulation window on your desktop, run:")
    print("    python scripts/run_milestone_03.py --gui")
    print("or:")
    print("    python scripts/view_live_simulation.py --mode arithmetic\n")

    print("=" * 70)
    print(" MILESTONE 3 EXPERIMENTAL EXECUTION COMPLETE")
    print("=" * 70)

    if "--gui" in sys.argv or "--view" in sys.argv:
        from scripts.view_live_simulation import view_arithmetic
        view_arithmetic()


if __name__ == "__main__":
    main()
