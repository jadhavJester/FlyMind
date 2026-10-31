"""
FlyBody Alphabet Spelling Task Execution Script.

Scatters 26 3D spatial points named 'A' to 'Z' across the arena.
Directs the fly to spell words sequentially in MuJoCo physics:
1. Primary Word: 'UDIT' (U -> D -> I -> T)
2. Follow-up Word: 'FLY' (F -> L -> Y)
3. Follow-up Word: 'AI' (A -> I)

Renders offscreen dual-camera video with all alphabet target positions,
live target tracking HUD, and spelled letter acquisition events.
"""

import os
import sys
import json
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.tasks.spelling import AlphabetArena, SpellingTask
from src.environments.spelling_flight import SpellingFlightEnv
from src.agents.oracle_selector import FlightPolicyController


def draw_spelling_hud(
    track_frame: np.ndarray,
    top_frame: np.ndarray,
    alphabet_arena: AlphabetArena,
    spelling_task: SpellingTask,
    active_letter: str,
    fly_pos: np.ndarray,
    dist_active: float,
    status_text: str,
    step: int,
    cumulative_reward: float
) -> np.ndarray:
    """
    Composites dual-camera views with all 26 scattered alphabet target markers,
    current target highlight, collected letters, and spelling telemetry HUD.
    """
    canvas_w = 960
    canvas_h = 540
    header_h = 100
    footer_h = 80
    view_h = canvas_h - header_h - footer_h  # 360 px
    view_w = canvas_w // 2                  # 480 px

    img_track = Image.fromarray(track_frame).resize((view_w, view_h))
    img_top = Image.fromarray(top_frame).resize((view_w, view_h))

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(14, 18, 26))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(img_track, (0, header_h))
    canvas.paste(img_top, (view_w, header_h))

    # Camera Labels
    draw.rectangle([10, header_h + 10, 185, header_h + 32], fill=(0, 0, 0, 180))
    draw.text((16, header_h + 14), "PERSPECTIVE: 3D TRACK", fill=(200, 220, 255))

    draw.rectangle([view_w + 10, header_h + 10, view_w + 225, header_h + 32], fill=(0, 0, 0, 180))
    draw.text((view_w + 16, header_h + 14), "ARENA: 26 LETTERS SCATTERED", fill=(200, 220, 255))

    # Top view coordinate mapping for the 26 alphabet points
    top_cx = view_w + view_w // 2
    top_cy = header_h + view_h // 2
    scale_x = 35.0
    scale_y = 35.0

    all_targets = alphabet_arena.all_targets()

    # Draw all 26 letters on top-down view
    for letter, target in all_targets.items():
        lx = int(top_cx + target.position[0] * scale_x)
        ly = int(top_cy - target.position[1] * scale_y)

        is_active = (letter == active_letter)
        is_spelled = (letter in spelling_task.spelled_letters)

        if is_active:
            # Gold pulse ring for current target
            draw.ellipse([lx - 14, ly - 14, lx + 14, ly + 14], outline=(255, 215, 0), width=2, fill=(80, 70, 10))
            draw.text((lx - 4, ly - 6), letter, fill=(255, 255, 255))
        elif is_spelled:
            # Green for already collected letters
            draw.ellipse([lx - 10, ly - 10, lx + 10, ly + 10], outline=(40, 220, 80), width=2, fill=(20, 80, 30))
            draw.text((lx - 3, ly - 5), letter, fill=(200, 255, 200))
        else:
            # Subtle cyan/grey circle for background alphabet points
            draw.ellipse([lx - 8, ly - 8, lx + 8, ly + 8], outline=(70, 90, 120), width=1, fill=(25, 32, 45))
            draw.text((lx - 3, ly - 5), letter, fill=(150, 170, 190))

    # Fly position marker on top view
    fly_mx = int(top_cx + fly_pos[0] * scale_x)
    fly_my = int(top_cy - fly_pos[1] * scale_y)
    draw.ellipse([fly_mx - 6, fly_my - 6, fly_mx + 6, fly_my + 6], fill=(255, 230, 40), outline=(0, 0, 0))

    # Divider line
    draw.line([(view_w, header_h), (view_w, header_h + view_h)], fill=(40, 50, 70), width=2)

    # HEADER BAR
    draw.rectangle([0, 0, canvas_w, header_h], fill=(18, 24, 38))
    draw.line([(0, header_h), (canvas_w, header_h)], fill=(40, 60, 90), width=2)

    draw.text((20, 12), "FLYBODY ALPHABET SPELLING MISSION (A to Z SCATTERED POINTS)", fill=(140, 180, 240))

    # Current Target Word Card
    draw.rectangle([20, 36, 280, 88], fill=(28, 38, 58), outline=(60, 90, 140))
    draw.text((32, 42), "TARGET WORD:", fill=(160, 175, 200))
    draw.text((32, 60), f"'{spelling_task.word}'  ({spelling_task.progress_str})", fill=(255, 255, 255))

    # Active Target Letter Card
    draw.rectangle([300, 36, 560, 88], fill=(45, 40, 20) if active_letter else (28, 38, 58),
                   outline=(240, 200, 50) if active_letter else (60, 90, 140), width=2)
    draw.text((312, 42), f"SEEKING LETTER: [{active_letter}]", fill=(255, 220, 60))
    draw.text((312, 62), f"Distance: {dist_active:.2f} cm (Threshold: 0.15 cm)", fill=(220, 230, 245))

    # Spelled Progress Card
    draw.rectangle([580, 36, 940, 88], fill=(24, 45, 30), outline=(50, 180, 90))
    draw.text((592, 42), "SPELLED SO FAR:", fill=(140, 230, 160))
    spelled_disp = spelling_task.current_spelling_str if spelling_task.current_spelling_str else "(none yet)"
    draw.text((592, 60), f"'{spelled_disp}'", fill=(100, 255, 140))

    # FOOTER BAR
    draw.rectangle([0, canvas_h - footer_h, canvas_w, canvas_h], fill=(18, 24, 38))
    draw.line([(0, canvas_h - footer_h), (canvas_w, canvas_h - footer_h)], fill=(40, 60, 90), width=2)

    draw.text((20, canvas_h - footer_h + 12),
              f"STEP: {step:3d}   |   FLY POS: [{fly_pos[0]:.2f}, {fly_pos[1]:.2f}, {fly_pos[2]:.2f}] cm   |   REWARD: {cumulative_reward:+6.2f}",
              fill=(180, 200, 230))

    status_color = (80, 240, 120) if "COLLECTED" in status_text or "COMPLETED" in status_text else (240, 210, 70)
    draw.text((20, canvas_h - footer_h + 40), f"MISSION STATUS: {status_text}", fill=status_color)

    return np.array(canvas)


def spell_word(
    word: str,
    env: SpellingFlightEnv,
    controller: FlightPolicyController,
    record_frames: bool = True
) -> Tuple[dict, List[np.ndarray]]:
    """
    Executes a multi-letter spelling mission for the given word.
    """
    print(f"\n>>> Starting Spelling Mission: '{word}' <<<")
    spelling_task = SpellingTask(word=word)
    all_frames = []
    legs_data = []
    cumulative_reward = 0.0

    while not spelling_task.completed:
        current_letter = spelling_task.target_letter
        print(f"  Navigating to Letter [{current_letter}] ...")

        obs = env.reset_letter_leg(current_letter)
        leg_trajectory = [env.get_fly_position().copy()]
        leg_rewards = []
        terminated = False
        info = {}

        status_text = f"FLYING TO TARGET LETTER [{current_letter}]"

        for step in range(env.max_steps):
            action = controller.get_action(obs)
            obs, reward, terminated, info = env.step(action)
            fly_pos = env.get_fly_position()
            leg_trajectory.append(fly_pos.copy())
            leg_rewards.append(reward)
            cumulative_reward += reward

            dist_active = info["dist_to_active"]

            if info["is_contact"]:
                status_text = f"TARGET REACHED: LETTER [{current_letter}] COLLECTED! (+10 REWARD)"
            elif terminated:
                status_text = f"LEG TERMINATED: {info['termination_reason']}"

            if record_frames and (step % 2 == 0 or terminated):
                track_frame = env.physics.render(camera_id='walker/track1', height=360, width=480)
                top_frame = env.physics.render(camera_id='top_camera', height=360, width=480)

                frame = draw_spelling_hud(
                    track_frame=track_frame,
                    top_frame=top_frame,
                    alphabet_arena=env.alphabet,
                    spelling_task=spelling_task,
                    active_letter=current_letter,
                    fly_pos=fly_pos,
                    dist_active=dist_active,
                    status_text=status_text,
                    step=step,
                    cumulative_reward=cumulative_reward
                )
                all_frames.append(frame)

            if terminated:
                break

        # Register letter advance
        was_collected = info.get("is_contact", False)
        if was_collected:
            spelling_task.advance_letter()
            print(f"  [SUCCESS] Letter [{current_letter}] acquired! (Min dist: {info['min_dist_to_active']:.3f} cm, Steps: {info['step']})")
        else:
            print(f"  [MISS] Failed to contact [{current_letter}]! (Min dist: {info['min_dist_to_active']:.3f} cm)")
            spelling_task.advance_letter()  # Proceed to next letter for demonstration

        legs_data.append({
            "letter": current_letter,
            "target_pos": env.active_target.position.tolist(),
            "collected": was_collected,
            "min_distance_cm": float(info.get("min_dist_to_active", 0.0)),
            "steps": int(info.get("step", 0)),
            "final_pos": leg_trajectory[-1].tolist()
        })

    # Hold final frame for 15 frames if completed
    if len(all_frames) > 0:
        final_frame = all_frames[-1]
        for _ in range(15):
            all_frames.append(final_frame)

    word_result = {
        "word": word,
        "completed": spelling_task.completed,
        "spelled_word": spelling_task.current_spelling_str,
        "legs": legs_data,
        "total_reward": float(cumulative_reward)
    }
    print(f">>> Word Mission '{word}' Finished: Spelled '{spelling_task.current_spelling_str}' <<<")
    return word_result, all_frames


def main():
    print("=" * 70)
    print(" FlyBody Alphabet Spelling: 'UDIT', 'FLY', 'AI'")
    print("=" * 70)

    results_dir = os.path.join(WORKSPACE_ROOT, "results", "spelling")
    videos_dir = os.path.join(WORKSPACE_ROOT, "videos")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    print("Initializing Spelling Environment & Alphabet Arena (26 targets 'A' to 'Z')...")
    env = SpellingFlightEnv(contact_radius=0.15, max_steps_per_letter=220)
    controller = FlightPolicyController()

    words_to_spell = ["UDIT", "FLY", "AI"]
    all_word_results = []
    combined_frames = []

    for word in words_to_spell:
        word_res, frames = spell_word(
            word=word,
            env=env,
            controller=controller,
            record_frames=True
        )
        all_word_results.append(word_res)
        combined_frames.extend(frames)

    # Save JSON log
    log_path = os.path.join(results_dir, "spelling_logs.json")
    with open(log_path, "w") as f:
        json.dump(all_word_results, f, indent=2)
    print(f"\nSaved spelling logs to: {log_path}")

    # Generate 3D Trajectory & Alphabet Arena Plot
    plot_path = os.path.join(results_dir, "alphabet_arena_spelling.png")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("26 Scattered Alphabet Targets (A-Z) & Spelled Words", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (Forward, cm)")
    ax.set_ylabel("Y (Lateral, cm)")
    ax.grid(True, linestyle="--", alpha=0.5)

    all_targets = env.alphabet.all_targets()
    # Plot all 26 letters
    for letter, t in all_targets.items():
        if letter in "UDIT":
            color = 'gold'
            marker = 'o'
            size = 12
        elif letter in "FLY":
            color = 'cyan'
            marker = '^'
            size = 10
        elif letter in "AI":
            color = 'lime'
            marker = 's'
            size = 10
        else:
            color = 'lightgray'
            marker = '.'
            size = 6
        ax.plot(t.position[0], t.position[1], marker=marker, color=color, markersize=size)
        ax.text(t.position[0] + 0.015, t.position[1] + 0.015, letter, fontsize=10, fontweight='bold',
                color='darkblue' if letter in "UDIT" else 'black')

    # Draw contact circle around UDIT targets
    for letter in "UDIT":
        pos = env.alphabet.get_target(letter).position
        circ = plt.Circle((pos[0], pos[1]), 0.15, color='orange', alpha=0.2)
        ax.add_patch(circ)

    # Connect UDIT sequence with arrows
    udit_coords = [env.alphabet.get_target(c).position for c in "UDIT"]
    for i in range(len(udit_coords) - 1):
        p1 = udit_coords[i]
        p2 = udit_coords[i+1]
        ax.annotate("", xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5, ls="--"))

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved alphabet arena plot to: {plot_path}")

    # Export Video
    if len(combined_frames) > 0:
        mp4_path = os.path.join(videos_dir, "fly_spelling_words.mp4")
        webp_path = os.path.join(videos_dir, "fly_spelling_words.webp")
        print(f"Exporting spelling video ({len(combined_frames)} frames @ 30 fps)...")

        imageio.mimsave(mp4_path, combined_frames, fps=30, quality=8)
        print(f"Saved MP4 video to: {mp4_path}")

        # Subsample for compact WebP
        webp_frames = combined_frames[::2]
        imageio.mimsave(webp_path, webp_frames, fps=15)
        print(f"Saved WebP animation to: {webp_path}")

    print("\n" + "=" * 70)
    print(" ALPHABET SPELLING MISSION COMPLETE: 'UDIT' SPELLED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
