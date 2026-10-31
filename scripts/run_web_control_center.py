"""
FlyBody Interactive Real-Time Web Control Center.

Features:
- Real-time live simulation video stream (dual-camera 3D tracking & arena top-down).
- Real-time task switching:
  1. Arithmetic Selector: Enter any A + B, fly computes and navigates to the spatial target answer.
  2. Alphabet Speller: Enter any word (e.g. 'UDIT', 'FLY', 'AI'), fly sequences through each letter.
  3. Direct 26-Letter Grid: Click any letter 'A'..'Z' to command the fly to that 3D waypoint.
  4. Manual Directional Steering: Steer Left, Right, Forward, Reset.
  5. Live Telemetry HUD: Coordinates (X, Y, Z), Speed, Target Distances, Contact Alerts.

Runs on local web server (http://localhost:8080) with zero external pip dependencies (standard library http.server).
"""

import os
import sys
import io
import time
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image, ImageDraw

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Auto-switch to workspace .venv if invoked with external/system Python
venv_python = os.path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([venv_python] + sys.argv))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import flybody
from flybody import fly_envs
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from src.tasks.arithmetic import ArithmeticTaskGenerator, DeterministicArithmeticSolver
from src.tasks.spelling import AlphabetArena, SpellingTask
from src.agents.oracle_selector import FlightPolicyController
from src.data.flight_dataset import quat_to_euler


class SimulationManager:
    """Manages the running FlyBody simulation and handles real-time task changes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.alphabet = AlphabetArena(contact_radius=0.15)
        self.controller = FlightPolicyController()

        # Lazy environment initialization: must be created on the simulation worker thread
        # so MuJoCo GLFW/OpenGL context is exclusively bound to that single thread.
        self.env = None
        self.dt = 0.0002

        # State
        self.active_mode = "arithmetic"  # "arithmetic", "spelling", "letter", "manual"
        self.active_task_desc = "2 + 3 = 5 -> target_left [5]"
        self.target_pos = np.array([0.60, 0.25, 0.55], dtype=np.float32)
        self.target_label = "[5]"
        self.target_name = "target_left"

        # Arithmetic state
        self.arith_task_gen = ArithmeticTaskGenerator()
        self.current_arith_task = self.arith_task_gen.generate_default_v01_task()

        # Spelling state
        self.spelling_task = SpellingTask(word="UDIT")
        self.spelling_active = False

        # Camera selection
        self.active_camera = "walker/track1"

        # Telemetry
        self.fly_pos = np.array([0.0, 0.0, 1.037])
        self.fly_vel = np.array([0.0, 0.0, 0.0])
        self.dist_to_target = 0.0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.mission_status = "INITIALIZING..."
        self.last_contact_letter = None

        # Queue for thread-safe command execution from HTTP thread to simulation worker thread
        self.pending_command = None
        self.is_running = True
        self.current_jpeg_bytes = None

    def ensure_env(self):
        """Initializes dm_control on the worker thread."""
        if self.env is None:
            self.env = fly_envs.flight_imitation(ref_path=None, terminal_com_dist=float('inf'))
            self.dt = self.env.control_timestep()
            self._reset_trajectory_to_target()
            self._render_current_frame_bytes()

    def _set_letter_target_internal(self, letter: str):
        target = self.alphabet.get_target(letter)
        if target is not None:
            self.target_pos = target.position.copy()
            self.target_label = f"[{letter}]"
            self.target_name = f"letter_{letter}"
            self._reset_trajectory_to_target()

    def _render_current_frame_bytes(self):
        if self.env is None:
            return
        try:
            track_frame = self.env.physics.render(camera_id='walker/track1', height=360, width=480)
            top_frame = self.env.physics.render(camera_id='top_camera', height=360, width=480)
            frame = self._render_stream_frame(track_frame, top_frame)
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=80)
            self.current_jpeg_bytes = buf.getvalue()
        except Exception as e:
            print(f"[RENDER ERROR]: {e}")

    def _apply_pending_command(self):
        if self.pending_command is None:
            return
        cmd = self.pending_command
        self.pending_command = None

        cmd_type = cmd[0]
        if cmd_type == "arithmetic":
            a, b = cmd[1], cmd[2]
            self.active_mode = "arithmetic"
            self.current_arith_task = self.arith_task_gen.create_task(
                operand_a=a,
                operand_b=b,
                correct_on_left=True,
                task_id=f"arith_{a}_plus_{b}"
            )
            decision = DeterministicArithmeticSolver.solve(self.current_arith_task)
            self.target_name = decision["selected_target_id"]
            self.target_label = f"[{decision['computed_answer']}]"
            if self.target_name == "target_left":
                self.target_pos = np.array([0.60, +0.25, 0.55], dtype=np.float32)
            else:
                self.target_pos = np.array([0.60, -0.25, 0.55], dtype=np.float32)

            self.active_task_desc = f"{a} + {b} = {decision['computed_answer']} -> {self.target_name} {self.target_label}"
            self.mission_status = f"SOLVED: {a} + {b} = {decision['computed_answer']}. NAVIGATING..."
            self._reset_trajectory_to_target()

        elif cmd_type == "spelling":
            word = cmd[1]
            clean_word = "".join([c.upper() for c in word if c.isalpha()])
            if not clean_word:
                clean_word = "UDIT"
            self.active_mode = "spelling"
            self.spelling_task = SpellingTask(word=clean_word)
            self.spelling_active = True
            first_letter = self.spelling_task.target_letter
            self._set_letter_target_internal(first_letter)
            self.active_task_desc = f"Spelling '{clean_word}': target [{first_letter}]"
            self.mission_status = f"SPELLING '{clean_word}': SEEKING [{first_letter}]"

        elif cmd_type == "letter":
            letter = cmd[1]
            self.active_mode = "letter"
            self.spelling_active = False
            self._set_letter_target_internal(letter.upper())
            self.active_task_desc = f"Direct Target: Letter [{letter.upper()}]"
            self.mission_status = f"NAVIGATING DIRECTLY TO LETTER [{letter.upper()}]"

        elif cmd_type == "steer":
            direction = cmd[1]
            self.active_mode = "manual"
            if direction == "left":
                self.target_pos = np.array([0.60, +0.35, 0.60], dtype=np.float32)
                self.target_label = "[LEFT]"
            elif direction == "right":
                self.target_pos = np.array([0.60, -0.35, 0.60], dtype=np.float32)
                self.target_label = "[RIGHT]"
            elif direction == "forward":
                self.target_pos = np.array([0.80, 0.00, 0.60], dtype=np.float32)
                self.target_label = "[FORWARD]"
            self.active_task_desc = f"Manual Command: {direction.upper()}"
            self.mission_status = f"MANUAL STEER: {direction.upper()}"
            self._reset_trajectory_to_target()

        elif cmd_type == "backflip":
            self.active_mode = "backflip"
            self.backflip_step = 0
            self.active_task_desc = "ACROBATIC MANEUVER: 360° BACKFLIP"
            self.target_label = "[BACKFLIP]"
            self.mission_status = "EXECUTING 360° PITCH LOOP..."

        elif cmd_type == "reset":
            self._reset_trajectory_to_target()
            self.cumulative_reward = 0.0
            self.mission_status = "RESET"

    def _reset_trajectory_to_target(self):
        if self.env is None:
            return
        heading = float(np.arctan2(self.target_pos[1], self.target_pos[0]))
        qp, qv = constant_speed_trajectory(
            n_steps=1500,
            speed=25.0,
            init_pos=(0.0, 0.0, 1.037),
            init_heading=heading,
            body_rot_angle_y=-47.5,
            control_timestep=self.dt
        )
        self.env.task._traj_generator.set_next_trajectory(qp, qv)
        self.env.reset()
        self.step_count = 0
        self.fly_pos = self.env.physics.named.data.xpos['walker/thorax'].copy()

    def set_arithmetic_task(self, a: int, b: int):
        with self.lock:
            self.pending_command = ("arithmetic", a, b)
            self.active_mode = "arithmetic"
            ans = a + b
            self.active_task_desc = f"{a} + {b} = {ans} -> target_left [{ans}]"
            self.target_label = f"[{ans}]"
            self.mission_status = f"SOLVED: {a} + {b} = {ans}. NAVIGATING..."

    def set_spelling_word(self, word: str):
        with self.lock:
            clean = "".join([c.upper() for c in word if c.isalpha()]) or "UDIT"
            self.pending_command = ("spelling", clean)
            self.active_mode = "spelling"
            self.active_task_desc = f"Spelling '{clean}': target [{clean[0]}]"
            self.target_label = f"[{clean[0]}]"
            self.mission_status = f"SPELLING '{clean}': SEEKING [{clean[0]}]"

    def set_letter_target(self, letter: str):
        with self.lock:
            l = letter.upper()
            self.pending_command = ("letter", l)
            self.active_mode = "letter"
            self.active_task_desc = f"Direct Target: Letter [{l}]"
            self.target_label = f"[{l}]"
            self.mission_status = f"NAVIGATING DIRECTLY TO LETTER [{l}]"

    def manual_steer(self, direction: str):
        with self.lock:
            self.pending_command = ("steer", direction)
            self.active_mode = "manual"
            self.active_task_desc = f"Manual Command: {direction.upper()}"
            self.target_label = f"[{direction.upper()}]"
            self.mission_status = f"MANUAL STEER: {direction.upper()}"

    def trigger_backflip(self):
        with self.lock:
            self.pending_command = ("backflip",)
            self.active_mode = "backflip"
            self.active_task_desc = "ACROBATIC MANEUVER: 360° BACKFLIP"
            self.target_label = "[BACKFLIP]"
            self.mission_status = "EXECUTING 360° PITCH LOOP..."

    def reset_sim(self):
        with self.lock:
            self.pending_command = ("reset",)
            self.cumulative_reward = 0.0
            self.mission_status = "RESET"

    def step_simulation(self):
        with self.lock:
            self.ensure_env()
            self._apply_pending_command()
            physics = self.env.physics
            fly_qvel = physics.data.qvel[:6]
            fly_lin_vel = fly_qvel[:3].copy()
            fly_ang_vel = fly_qvel[3:6].copy()

            fly_qpos = physics.data.qpos[:7]
            self.fly_pos = fly_qpos[:3].copy()
            self.fly_vel = fly_lin_vel
            fly_quat = fly_qpos[3:7].copy()
            fly_euler = quat_to_euler(fly_quat)

            # Lookahead error vector
            traj_gen = self.env.task._traj_generator
            com_qpos = getattr(traj_gen, '_com_qpos', None)
            cur_step = self.step_count

            future_disp = []
            for k in range(1, 6):
                target_step = cur_step + k
                if com_qpos is not None and isinstance(com_qpos, np.ndarray) and target_step < com_qpos.shape[0]:
                    ref_point = com_qpos[target_step, :3]
                else:
                    ref_point = self.target_pos
                future_disp.extend(ref_point - self.fly_pos)

            raw_feature = np.concatenate([
                fly_lin_vel,
                fly_ang_vel,
                fly_quat,
                fly_euler,
                np.array(future_disp, dtype=np.float32)
            ]).astype(np.float32)

            # Policy or Acrobatic Action
            if self.active_mode == "backflip":
                self.backflip_step = getattr(self, "backflip_step", 0) + 1
                if self.backflip_step < 15:
                    action = np.zeros(12, dtype=np.float32)
                    action[3] = 0.1; action[6] = 0.1; action[11] = 0.5
                elif self.backflip_step < 133:
                    action = np.zeros(12, dtype=np.float32)
                    action[2] = -0.3; action[3] = 0.9; action[5] = 0.7; action[6] = 0.9; action[8] = 0.7; action[10] = 0.7; action[11] = 1.0
                elif self.backflip_step < 158:
                    action = np.zeros(12, dtype=np.float32)
                    action[2] = 0.2; action[3] = -0.7; action[5] = -0.4; action[6] = -0.7; action[8] = -0.4; action[10] = -0.8; action[11] = 0.8
                else:
                    self.active_mode = "manual"
                    self.mission_status = "BACKFLIP COMPLETE! LEVEL FLIGHT."
                    action = np.zeros(12, dtype=np.float32)
                    action[3] = 0.1; action[6] = 0.1; action[11] = 0.5
            else:
                action = self.controller.policy.select_action((raw_feature - self.controller.mean) / self.controller.std)
            ts = self.env.step(action)
            self.step_count += 1

            self.fly_pos = physics.named.data.xpos['walker/thorax'].copy()
            self.dist_to_target = float(np.linalg.norm(self.fly_pos - self.target_pos))

            # Contact detection (r <= 0.40 cm contact bubble for 3cm spaced targets)
            if self.dist_to_target <= 0.40:
                self.cumulative_reward += 10.0
                if self.active_mode == "spelling" and self.spelling_active:
                    current_l = self.spelling_task.target_letter
                    self.last_contact_letter = current_l
                    self.spelling_task.advance_letter()
                    if self.spelling_task.completed:
                        self.mission_status = f"MISSION COMPLETE: SPELLED '{self.spelling_task.word}'! (+10)"
                        self.spelling_active = False
                    else:
                        next_l = self.spelling_task.target_letter
                        self.mission_status = f"COLLECTED [{current_l}]! NEXT: [{next_l}]"
                        self._set_letter_target_internal(next_l)
                else:
                    self.mission_status = f"TARGET REACHED {self.target_label}! (+10 REWARD)"
                    time.sleep(0.3)
                    self._reset_trajectory_to_target()

            elif self.fly_pos[2] < 0.20 or self.step_count >= 1200:
                self.mission_status = "TOUCHDOWN / RE-TRIMMING FOR NEXT RUN"
                self._reset_trajectory_to_target()

            # Render composite HUD frame for stream
            self._render_current_frame_bytes()

    def _render_stream_frame(self, track_frame, top_frame):
        canvas_w = 960
        canvas_h = 540
        header_h = 70
        footer_h = 60
        view_h = canvas_h - header_h - footer_h
        view_w = canvas_w // 2

        img_track = Image.fromarray(track_frame).resize((view_w, view_h))
        img_top = Image.fromarray(top_frame).resize((view_w, view_h))

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(11, 15, 25))
        draw = ImageDraw.Draw(canvas)

        canvas.paste(img_track, (0, header_h))
        canvas.paste(img_top, (view_w, header_h))

        # Top view coordinate mapping for letters & target (Arena top-down radar)
        radar_cx = view_w + view_w // 2
        radar_cy = header_h + view_h // 2
        scale = 36.0  # pixels per cm

        # Draw scattered letters with large ~3cm separation
        for letter, target in self.alphabet.all_targets().items():
            # X is forward (upwards), Y is lateral (left/right)
            lx = int(radar_cx + target.position[1] * scale)
            ly = int(radar_cy - (target.position[0] - 5.0) * scale)
            is_active = (letter == self.target_label.strip("[]"))
            is_spelled = (letter in self.spelling_task.spelled_letters) if self.active_mode == "spelling" else False

            if is_active:
                draw.ellipse([lx - 14, ly - 14, lx + 14, ly + 14], outline=(255, 215, 0), width=3, fill=(100, 85, 15))
                draw.text((lx - 5, ly - 7), letter, fill=(255, 255, 255))
            elif is_spelled:
                draw.ellipse([lx - 10, ly - 10, lx + 10, ly + 10], outline=(0, 230, 118), width=2, fill=(20, 80, 30))
                draw.text((lx - 4, ly - 6), letter, fill=(180, 255, 200))
            else:
                draw.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], outline=(70, 95, 135), width=1, fill=(24, 32, 50))
                draw.text((lx - 4, ly - 6), letter, fill=(160, 185, 210))

        # Draw Fly marker
        fly_mx = int(radar_cx + self.fly_pos[1] * scale)
        fly_my = int(radar_cy - (self.fly_pos[0] - 5.0) * scale)
        draw.ellipse([fly_mx - 6, fly_my - 6, fly_mx + 6, fly_my + 6], fill=(255, 235, 59), outline=(0, 0, 0), width=2)

        draw.line([(view_w, header_h), (view_w, header_h + view_h)], fill=(40, 50, 75), width=2)

        # Header
        draw.rectangle([0, 0, canvas_w, header_h], fill=(16, 22, 36))
        draw.text((20, 10), "FLYBODY REAL-TIME EMBODIED TASK CONTROL CENTER", fill=(0, 210, 255))
        draw.text((20, 32), f"ACTIVE TASK: {self.active_task_desc}", fill=(255, 255, 255))
        draw.text((640, 32), f"TARGET: {self.target_label} (Dist: {self.dist_to_target:.2f} cm)", fill=(255, 215, 0))

        # Footer
        draw.rectangle([0, canvas_h - footer_h, canvas_w, canvas_h], fill=(16, 22, 36))
        draw.text((20, canvas_h - footer_h + 12),
                  f"FLY POS: [{self.fly_pos[0]:.2f}, {self.fly_pos[1]:.2f}, {self.fly_pos[2]:.2f}] cm   |   "
                  f"SPEED: {np.linalg.norm(self.fly_vel):.1f} cm/s   |   STEP: {self.step_count}   |   REWARD: {self.cumulative_reward:+.1f}",
                  fill=(180, 200, 230))
        status_col = (0, 230, 118) if "REACHED" in self.mission_status or "COMPLETE" in self.mission_status else (255, 200, 50)
        draw.text((20, canvas_h - footer_h + 34), f"STATUS: {self.mission_status}", fill=status_col)

        return canvas

    def get_telemetry_dict(self):
        with self.lock:
            return {
                "active_mode": self.active_mode,
                "active_task_desc": self.active_task_desc,
                "target_label": self.target_label,
                "target_pos": self.target_pos.tolist(),
                "dist_to_target": float(self.dist_to_target),
                "fly_pos": self.fly_pos.tolist(),
                "fly_speed": float(np.linalg.norm(self.fly_vel)),
                "step_count": int(self.step_count),
                "cumulative_reward": float(self.cumulative_reward),
                "mission_status": self.mission_status,
                "spelling_word": self.spelling_task.word if self.spelling_task else "",
                "spelled_so_far": self.spelling_task.current_spelling_str if self.spelling_task else "",
                "spelling_completed": self.spelling_task.completed if self.spelling_task else False
            }


# Global Simulation Manager instance
SIM_MANAGER = SimulationManager()


def simulation_worker():
    """Runs physics simulation updates at ~25-30 Hz for the live stream."""
    print("[SIMULATION WORKER] Background physics simulation loop running...")
    while getattr(SIM_MANAGER, "is_running", True):
        t0 = time.time()
        try:
            SIM_MANAGER.step_simulation()
        except Exception as e:
            print(f"[SIMULATION STEP ERROR]: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
        elapsed = time.time() - t0
        sleep_time = max(0.005, 0.035 - elapsed)
        time.sleep(sleep_time)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FlyBody Real-Time Control Center</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(18, 25, 42, 0.85);
      --border: rgba(45, 60, 95, 0.6);
      --accent-cyan: #00d2ff;
      --accent-green: #00e676;
      --accent-gold: #ffd600;
      --accent-coral: #ff5252;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 16px; min-height: 100vh; display: flex; flex-direction: column; gap: 16px; }
    
    header {
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 12px 20px;
      border-radius: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      backdrop-filter: blur(8px);
    }
    header h1 { font-size: 1.25rem; font-weight: 700; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px; }
    .badge { background: rgba(0, 210, 255, 0.15); color: var(--accent-cyan); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(0, 210, 255, 0.3); }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
      flex: 1;
    }

    .video-panel {
      background: #000;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .video-feed { width: 100%; height: auto; display: block; border-bottom: 1px solid var(--border); }

    .telemetry-bar {
      padding: 12px 16px;
      background: var(--card-bg);
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .metric-card {
      background: rgba(10, 15, 26, 0.6);
      border: 1px solid rgba(45, 60, 95, 0.4);
      padding: 8px 12px;
      border-radius: 8px;
    }
    .metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-val { font-size: 1.05rem; font-weight: 700; color: #fff; margin-top: 2px; }

    .control-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      backdrop-filter: blur(8px);
    }
    .card-title { font-size: 0.9rem; font-weight: 700; color: var(--accent-gold); display: flex; align-items: center; justify-content: space-between; }
    
    .input-row { display: flex; gap: 8px; align-items: center; }
    input[type="number"], input[type="text"] {
      background: rgba(10, 15, 26, 0.8);
      border: 1px solid var(--border);
      color: #fff;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 0.95rem;
      flex: 1;
      outline: none;
    }
    input:focus { border-color: var(--accent-cyan); }

    button {
      background: linear-gradient(135deg, #0077b6, #00b4d8);
      border: none;
      color: #fff;
      padding: 9px 14px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    button:hover { filter: brightness(1.15); transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    button.btn-accent { background: linear-gradient(135deg, #00b4d8, #00e676); }
    button.btn-danger { background: linear-gradient(135deg, #e63946, #d90429); }

    .alphabet-grid {
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 5px;
    }
    .btn-letter {
      padding: 6px 0;
      font-size: 0.85rem;
      font-weight: 700;
      background: rgba(25, 35, 55, 0.7);
      border: 1px solid rgba(60, 80, 120, 0.5);
      border-radius: 6px;
      color: #cbd5e1;
    }
    .btn-letter:hover {
      background: var(--accent-cyan);
      color: #000;
      border-color: #fff;
    }

    .status-alert {
      padding: 8px 12px;
      background: rgba(0, 230, 118, 0.1);
      border: 1px solid rgba(0, 230, 118, 0.3);
      color: var(--accent-green);
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
    }
  </style>
</head>
<body>

  <header>
    <h1>🪰 FlyBody Real-Time Task Controller</h1>
    <div style="display:flex; gap:8px; align-items:center;">
      <span class="badge" id="mode-badge">MODE: ARITHMETIC</span>
      <button onclick="resetSimulation()" class="btn-danger" style="padding:6px 12px; font-size:0.75rem;">Reset Fly</button>
    </div>
  </header>

  <div class="main-grid">
    <!-- Left: Video & Telemetry -->
    <div class="video-panel">
      <img id="live-stream-img" src="/api/current_frame.jpg" class="video-feed" alt="Live MuJoCo Stream">
      <div class="telemetry-bar">
        <div class="metric-card">
          <div class="metric-label">Active Target</div>
          <div class="metric-val" id="val-target" style="color:var(--accent-gold);">[5]</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Target Dist</div>
          <div class="metric-val" id="val-dist">0.00 cm</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Fly Position</div>
          <div class="metric-val" id="val-pos">[0.0, 0.0, 1.0]</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Reward</div>
          <div class="metric-val" id="val-reward" style="color:var(--accent-green);">+0.0</div>
        </div>
      </div>
    </div>

    <!-- Right: Interactive Task Switcher -->
    <div class="control-panel">

      <!-- Arithmetic Task Switcher -->
      <div class="card">
        <div class="card-title">
          <span>1. Arithmetic Problem Generator</span>
          <span style="font-size:0.75rem; color:var(--text-muted);">Real-Time Solver</span>
        </div>
        <div class="input-row">
          <input type="number" id="num-a" value="2" min="0" max="20" style="text-align:center;">
          <span style="font-weight:700; font-size:1.1rem;">+</span>
          <input type="number" id="num-b" value="3" min="0" max="20" style="text-align:center;">
          <button onclick="sendArithmetic()" class="btn-accent">Solve & Fly</button>
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted);">Fly computes sum and steers to matching spatial target.</div>
      </div>

      <!-- Word Speller Task Switcher -->
      <div class="card">
        <div class="card-title">
          <span>2. Alphabet Word Speller</span>
          <span style="font-size:0.75rem; color:var(--accent-green);">Sequential 3D Legs</span>
        </div>
        <div class="input-row">
          <input type="text" id="spell-word" value="UDIT" placeholder="Type word (e.g. UDIT, FLY, AI)">
          <button onclick="sendSpelling()">Spell Word</button>
        </div>
        <div id="spelling-progress" class="status-alert">Progress: [U] [D] [I] [T]</div>
      </div>

      <!-- Clickable Alphabet Grid (A-Z) -->
      <div class="card">
        <div class="card-title">
          <span>3. Click Any Letter to Steer (A to Z)</span>
          <span style="font-size:0.75rem; color:var(--text-muted);">26 Arena Points</span>
        </div>
        <div class="alphabet-grid" id="alphabet-buttons"></div>
      </div>

      <!-- Manual Steering Controls & Acrobatic Stunts -->
      <div class="card">
        <div class="card-title">
          <span>4. Quick Controls & Acrobatic Stunts</span>
          <span style="font-size:0.75rem; color:#f72585;">Acrobatics</span>
        </div>
        <div style="display:flex; gap:8px;">
          <button onclick="sendSteer('left')" style="flex:1;">Steer Left</button>
          <button onclick="sendSteer('forward')" style="flex:1;">Forward</button>
          <button onclick="sendSteer('right')" style="flex:1;">Steer Right</button>
        </div>
        <button onclick="sendBackflip()" style="background: linear-gradient(135deg, #f72585, #7209b7); width: 100%; margin-top: 8px; font-weight: 700; font-size: 0.95rem; box-shadow: 0 4px 15px rgba(247, 37, 133, 0.4);">🔄 Perform 360° Backflip</button>
      </div>

    </div>
  </div>

  <script>
    // Build 26 Alphabet Buttons
    const grid = document.getElementById('alphabet-buttons');
    for (let c = 65; c <= 90; c++) {
      const char = String.fromCharCode(c);
      const btn = document.createElement('button');
      btn.className = 'btn-letter';
      btn.innerText = char;
      btn.onclick = () => sendLetter(char);
      grid.appendChild(btn);
    }

    async function sendBackflip() {
      await fetch('/api/set_task', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: 'backflip' })
      });
    }

    async function sendArithmetic() {
      const a = parseInt(document.getElementById('num-a').value) || 2;
      const b = parseInt(document.getElementById('num-b').value) || 3;
      await fetch('/api/set_task', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: 'arithmetic', a: a, b: b })
      });
    }

    async function sendSpelling() {
      const word = document.getElementById('spell-word').value.trim() || 'UDIT';
      await fetch('/api/set_task', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: 'spelling', word: word })
      });
    }

    async function sendLetter(letter) {
      await fetch('/api/set_task', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: 'letter', letter: letter })
      });
    }

    async function sendSteer(dir) {
      await fetch('/api/set_task', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: 'steer', direction: dir })
      });
    }

    async function resetSimulation() {
      await fetch('/api/set_task', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: 'reset' })
      });
    }

    // Telemetry Poller
    setInterval(async () => {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        document.getElementById('val-target').innerText = data.target_label;
        document.getElementById('val-dist').innerText = data.dist_to_target.toFixed(2) + ' cm';
        document.getElementById('val-pos').innerText = `[${data.fly_pos[0].toFixed(2)}, ${data.fly_pos[1].toFixed(2)}, ${data.fly_pos[2].toFixed(2)}]`;
        document.getElementById('val-reward').innerText = (data.cumulative_reward >= 0 ? '+' : '') + data.cumulative_reward.toFixed(1);
        document.getElementById('mode-badge').innerText = 'MODE: ' + data.active_mode.toUpperCase();
        
        if (data.active_mode === 'spelling') {
          const prog = data.spelled_so_far ? `Spelled: '${data.spelled_so_far}' / '${data.spelling_word}'` : `Seeking 1st letter of '${data.spelling_word}'`;
          document.getElementById('spelling-progress').innerText = prog;
        } else {
          document.getElementById('spelling-progress').innerText = data.mission_status;
        }
      } catch (e) {}
    }, 150);

    // High-speed smooth live video frame updater
    const liveImg = document.getElementById('live-stream-img');
    let isFetchingFrame = false;
    function fetchNextFrame() {
      if (isFetchingFrame) return;
      isFetchingFrame = true;
      const nextImg = new Image();
      nextImg.onload = function() {
        liveImg.src = nextImg.src;
        isFetchingFrame = false;
        setTimeout(fetchNextFrame, 35); // ~28-30 FPS
      };
      nextImg.onerror = function() {
        isFetchingFrame = false;
        setTimeout(fetchNextFrame, 100);
      };
      nextImg.src = '/api/current_frame.jpg?t=' + Date.now();
    }
    fetchNextFrame();
  </script>
</body>
</html>
"""


class WebRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for dashboard UI, telemetry, and live MJPEG stream."""

    def log_message(self, format, *args):
        # Silence default request logging for high-speed polling
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            resp = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif self.path == "/api/telemetry":
            telemetry = SIM_MANAGER.get_telemetry_dict()
            resp = json.dumps(telemetry).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif self.path.startswith("/api/current_frame.jpg"):
            frame_bytes = SIM_MANAGER.current_jpeg_bytes
            if frame_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(frame_bytes)
                self.wfile.flush()
            else:
                self.send_response(204)
                self.end_headers()

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while SIM_MANAGER.is_running:
                    frame_bytes = SIM_MANAGER.current_jpeg_bytes
                    if frame_bytes:
                        header = (
                            b"--FRAME\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                        )
                        self.wfile.write(header + frame_bytes + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.04)
            except (ConnectionResetError, BrokenPipeError):
                pass

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/set_task":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)

            cmd_type = data.get("type")
            if cmd_type == "arithmetic":
                a = int(data.get("a", 2))
                b = int(data.get("b", 3))
                SIM_MANAGER.set_arithmetic_task(a, b)
            elif cmd_type == "spelling":
                word = data.get("word", "UDIT")
                SIM_MANAGER.set_spelling_word(word)
            elif cmd_type == "letter":
                letter = data.get("letter", "U")
                SIM_MANAGER.set_letter_target(letter)
            elif cmd_type == "steer":
                direction = data.get("direction", "forward")
                SIM_MANAGER.manual_steer(direction)
            elif cmd_type == "backflip":
                SIM_MANAGER.trigger_backflip()
            elif cmd_type == "reset":
                SIM_MANAGER.reset_sim()

            resp = json.dumps({"status": "ok", "cmd": cmd_type}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8080
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, WebRequestHandler)

    print("=" * 70)
    print(" FLYBODY INTERACTIVE REAL-TIME TASK CONTROL CENTER")
    print("=" * 70)
    print(f"\n[LIVE SERVER] Running at: http://localhost:{port}")
    print(f"Open your browser at: http://localhost:{port}")
    print("\nFeatures:")
    print("  1. Watch live 3D flight simulation with dual-camera perspective.")
    print("  2. Dynamically change arithmetic tasks (e.g. 4 + 5, 1 + 8).")
    print("  3. Dynamically spell words (e.g. UDIT, FLY, AI, HELLO).")
    print("  4. Click any of the 26 letters (A to Z) to steer to that target in real-time.")
    print("  5. Manual left/right/forward steering buttons.\n")
    print("Press Ctrl+C to stop the server.")

    # Start HTTP web server in a daemon thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    # Main thread runs the simulation and OpenGL/WGL rendering loop
    # (required on Windows for GLFW context creation)
    try:
        simulation_worker()
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping Control Center...")
        SIM_MANAGER.is_running = False
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
        print("Control Center stopped cleanly.")


if __name__ == "__main__":
    main()
