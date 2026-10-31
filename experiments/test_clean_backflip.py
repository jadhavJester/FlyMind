"""
Experiment to tune a clean 360-degree backflip with recovery and stabilization.
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
from src.data.flight_dataset import quat_to_euler


def run_tuned_backflip():
    env = flight_imitation(terminal_com_dist=float('inf'))
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')
    controller = FlightPolicyController()

    dt = env.control_timestep()
    
    # Backflip pulse action
    backflip_action = np.array([
        0.0,   # head_abduct
        0.0,   # head_twist
        -0.2,  # head pitch
        0.85,  # wing_yaw_left
        0.0,   # wing_roll_left
        0.6,   # wing_pitch_left
        0.85,  # wing_yaw_right
        0.0,   # wing_roll_right
        0.6,   # wing_pitch_right
        0.0,   # abdomen_abduct
        0.6,   # abdomen dorsal flexion
        1.0,   # frequency boost
    ], dtype=np.float32)

    # Recovery brake action to arrest pitch rotation
    recovery_brake = np.array([
        0.0,
        0.0,
        0.2,
        -0.5,  # stroke backward to pitch down
        0.0,
        -0.4,
        -0.5,
        0.0,
        -0.4,
        0.0,
        -0.7,  # abdomen ventral to push nose down
        0.5,
    ], dtype=np.float32)

    # Let's test different pulse durations
    for pulse_steps in [60, 65, 70, 75, 80]:
        env.reset()
        init_xmat = env.physics.data.xmat[thorax_id].reshape(3, 3).copy()
        prev_fwd = init_xmat[:, 0].copy()
        total_pitch_deg = 0.0
        
        entry_steps = 30
        brake_steps = 25
        total_steps = 200
        
        records = []
        for step in range(total_steps):
            if step < entry_steps:
                # Normal cruise flight
                obs_vec = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
                act = controller.policy.select_action(obs_vec)
            elif step < entry_steps + pulse_steps:
                act = backflip_action
            elif step < entry_steps + pulse_steps + brake_steps:
                act = recovery_brake
            else:
                obs_vec = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
                act = controller.policy.select_action(obs_vec)

            env.step(act)
            
            xmat = env.physics.data.xmat[thorax_id].reshape(3, 3).copy()
            fwd = xmat[:, 0]
            lat = xmat[:, 1]
            pos = env.physics.data.xpos[thorax_id].copy()
            
            dot = np.clip(np.dot(prev_fwd, fwd), -1.0, 1.0)
            d_angle = np.arccos(dot)
            cross_y = np.cross(prev_fwd, fwd)
            sign = 1.0 if np.dot(cross_y, lat) > 0 else -1.0
            total_pitch_deg += np.rad2deg(sign * d_angle)
            prev_fwd = fwd.copy()
            
            records.append((step, pos[0], pos[2], total_pitch_deg))

        rec_arr = np.array(records)
        max_loop = np.max(rec_arr[:, 3])
        final_pitch = rec_arr[-1, 3]
        final_z = rec_arr[-1, 2]
        min_z = np.min(rec_arr[:, 2])
        print(f"Pulse: {pulse_steps:2d} steps | Max Pitch Rot: {max_loop:5.1f}° | Final Rot: {final_pitch:5.1f}° | Min Z: {min_z:5.2f} cm | Final Z: {final_z:5.2f} cm")

if __name__ == "__main__":
    run_tuned_backflip()
