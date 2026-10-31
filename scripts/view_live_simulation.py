"""
Interactive Live 3D Simulation Viewer for FlyBody.

Opens a native MuJoCo GLFW 3D window on your desktop to view the fly simulation live!

Interactive Controls in Viewer Window:
- Left-click + drag: Rotate camera
- Right-click + drag: Pan camera
- Mouse scroll wheel: Zoom in / out
- Double-click on fly: Track camera on the fly body
- Space: Pause / Resume simulation
- Backspace: Reset simulation

Usage:
    python scripts/view_live_simulation.py
    python scripts/view_live_simulation.py --mode milestone_01
    python scripts/view_live_simulation.py --mode arithmetic
    python scripts/view_live_simulation.py --mode spelling --word UDIT
"""

import os
import sys
import time
import argparse
import numpy as np

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

import mujoco
import mujoco.viewer
import flybody
from flybody import fly_envs
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from src.agents.oracle_selector import FlightPolicyController
from evaluation.run_evaluation import extract_features


def view_milestone_01(slowdown: float = 1.0):
    """Launches live 3D viewer for Milestone 1 flight simulation."""
    print("\n[LIVE VIEWER] Launching Milestone 1 Flight Simulation in interactive 3D window...")
    print("  Controls: Left-drag to rotate, Right-drag to pan, Scroll to zoom, Space to pause.")

    env = fly_envs.flight_imitation(ref_path=None, terminal_com_dist=float('inf'))
    env.reset()
    controller = FlightPolicyController()

    m = env.physics.model.ptr
    d = env.physics.data.ptr

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Initial camera tracking
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        viewer.cam.lookat[:] = [0.2, 0.0, 0.8]

        step = 0
        while viewer.is_running():
            step_start = time.time()

            obs_vec = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
            action = controller.policy.select_action(obs_vec)
            ts = env.step(action)
            step += 1

            # Update camera tracking target to follow fly
            fly_pos = env.physics.named.data.xpos['walker/thorax']
            viewer.cam.lookat[:] = fly_pos

            # Sync physics state to visual window
            viewer.sync()

            if ts.last() or step >= 500:
                print("  Loop reset: restarting flight path...")
                env.reset()
                step = 0

            # Real-time pacing (FlyBody dt is 0.0002s, slow down by factor for visibility)
            time_until_next = (env.control_timestep() * slowdown) - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


def view_arithmetic(slowdown: float = 1.0):
    """Launches live 3D viewer for Milestone 3 Drosophila Arithmetic Selector."""
    from src.tasks.arithmetic import ArithmeticTaskGenerator, DeterministicArithmeticSolver
    from src.environments.interactive_flight import InteractiveArithmeticEnv

    print("\n[LIVE VIEWER] Launching Milestone 3 Arithmetic Selector (2 + 3 = 5)...")
    task_gen = ArithmeticTaskGenerator()
    task = task_gen.generate_default_v01_task()
    decision = DeterministicArithmeticSolver.solve(task)
    selected_target = decision["selected_target_id"]
    print(f"  Task: {task.prompt_str} -> Answer: {decision['computed_answer']} (Target: {selected_target})")

    env = InteractiveArithmeticEnv(max_steps=300)
    controller = FlightPolicyController()

    obs = env.reset(task=task, selected_target_id=selected_target)
    m = env.physics.model.ptr
    d = env.physics.data.ptr

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 45

        step = 0
        while viewer.is_running():
            step_start = time.time()
            action = controller.get_action(obs)
            obs, reward, terminated, info = env.step(action)
            step += 1

            fly_pos = env.get_fly_position()
            viewer.cam.lookat[:] = fly_pos
            viewer.sync()

            if terminated:
                status = "SUCCESS (+10)" if info["success"] else info["termination_reason"]
                print(f"  Target Leg Complete! Status: {status} at step {step}. Restarting...")
                time.sleep(0.5)
                obs = env.reset(task=task, selected_target_id=selected_target)
                step = 0

            time_until_next = (env.dt * slowdown) - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


def view_spelling(word: str = "UDIT", slowdown: float = 1.0):
    """Launches live 3D viewer for Alphabet Spelling mission."""
    from src.tasks.spelling import SpellingTask
    from src.environments.spelling_flight import SpellingFlightEnv

    print(f"\n[LIVE VIEWER] Launching Alphabet Spelling Mission for word: '{word}'...")
    spelling_task = SpellingTask(word=word)
    env = SpellingFlightEnv(max_steps_per_letter=250)
    controller = FlightPolicyController()

    m = env.physics.model.ptr
    d = env.physics.data.ptr

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 60

        while viewer.is_running() and not spelling_task.completed:
            letter = spelling_task.target_letter
            print(f"  Seeking Letter: [{letter}] ...")
            obs = env.reset_letter_leg(letter)

            step = 0
            while viewer.is_running():
                step_start = time.time()
                action = controller.get_action(obs)
                obs, reward, terminated, info = env.step(action)
                step += 1

                fly_pos = env.get_fly_position()
                viewer.cam.lookat[:] = fly_pos
                viewer.sync()

                if terminated:
                    if info["is_contact"]:
                        spelling_task.advance_letter()
                        print(f"  >>> Letter [{letter}] COLLECTED! Progress: '{spelling_task.current_spelling_str}' <<<")
                    else:
                        spelling_task.advance_letter()
                    time.sleep(0.4)
                    break

                time_until_next = (env.dt * slowdown) - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        print(f"\n[MISSION COMPLETE] Word '{spelling_task.word}' spelled: '{spelling_task.current_spelling_str}'!")
        time.sleep(1.0)


def view_backflip(slowdown: float = 2.0):
    """Launches live 3D desktop window for Drosophila 360-degree backflip."""
    from scripts.fly_backflip import run_backflip_simulation
    run_backflip_simulation(launch_gui=True, render_video=False)


def main():
    parser = argparse.ArgumentParser(description="Live 3D MuJoCo Viewer for FlyBody")
    parser.add_argument("--mode", type=str, default="milestone_01",
                        choices=["milestone_01", "arithmetic", "spelling", "backflip"],
                        help="Simulation mode to view in 3D desktop window")
    parser.add_argument("--word", type=str, default="UDIT",
                        help="Word to spell when mode=spelling")
    parser.add_argument("--slowdown", type=float, default=2.0,
                        help="Visual slowdown multiplier (e.g. 2.0 = 2x slower for easy viewing)")
    args = parser.parse_args()

    if args.mode == "milestone_01":
        view_milestone_01(slowdown=args.slowdown)
    elif args.mode == "arithmetic":
        view_arithmetic(slowdown=args.slowdown)
    elif args.mode == "spelling":
        view_spelling(word=args.word, slowdown=args.slowdown)
    elif args.mode == "backflip":
        view_backflip(slowdown=args.slowdown)


if __name__ == "__main__":
    main()
