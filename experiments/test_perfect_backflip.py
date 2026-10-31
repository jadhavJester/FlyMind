"""
Verify exact 360-degree backflip with precise brake and recovery.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from src.agents.oracle_selector import FlightPolicyController
from evaluation.run_evaluation import extract_features

def test_perfect_backflip():
    env = flight_imitation(terminal_com_dist=float('inf'))
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')
    controller = FlightPolicyController()
    dt = env.control_timestep()

    # Backflip action
    loop_act = np.zeros(12, dtype=np.float32)
    loop_act[2] = -0.3   # head nod up
    loop_act[3] = 0.9    # wing_yaw_left forward sweep
    loop_act[5] = 0.7    # wing_pitch_left high AoA
    loop_act[6] = 0.9    # wing_yaw_right forward sweep
    loop_act[8] = 0.7    # wing_pitch_right high AoA
    loop_act[10] = 0.7   # abdomen dorsal flexion
    loop_act[11] = 1.0   # frequency boost

    # Counter-pitch brake action
    brake_act = np.zeros(12, dtype=np.float32)
    brake_act[2] = 0.2
    brake_act[3] = -0.5
    brake_act[5] = -0.3
    brake_act[6] = -0.5
    brake_act[8] = -0.3
    brake_act[10] = -0.8
    brake_act[11] = 0.5

    for loop_steps in [105, 110, 112, 115]:
        env.reset()
        rot = 0.0
        entry_steps = 30
        brake_steps = 30
        total_steps = 220

        trajectory = []
        for step in range(total_steps):
            if step < entry_steps:
                obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
                act = controller.policy.select_action(obs)
            elif step < entry_steps + loop_steps:
                act = loop_act
            elif step < entry_steps + loop_steps + brake_steps:
                act = brake_act
            else:
                obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
                act = controller.policy.select_action(obs)

            env.step(act)
            rot += env.physics.data.qvel[4] * dt
            z = env.physics.data.xpos[thorax_id, 2]
            trajectory.append((step, np.rad2deg(rot), z))

        arr = np.array(trajectory)
        peak_rot = np.max(arr[:, 1])
        final_rot = arr[-1, 1]
        min_z = np.min(arr[:, 2])
        final_z = arr[-1, 2]
        print(f"Loop Steps: {loop_steps:3d} | Peak Rot: {peak_rot:6.1f}° | Final Rot: {final_rot:6.1f}° | Min Z: {min_z:.2f} cm | Final Z: {final_z:.2f} cm")

if __name__ == "__main__":
    test_perfect_backflip()
