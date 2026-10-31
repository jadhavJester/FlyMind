"""
Experiment: Testing Backflip (Sagittal Pitch Loop) Maneuvers in FlyBody.
Investigates how the fruit fly can execute a complete 360-degree backflip.
"""

import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from src.data.flight_dataset import quat_to_euler


def test_pitch_maneuvers():
    env = flight_imitation(terminal_com_dist=float('inf'))
    
    # Let's test different pitch-inducing action profiles
    test_cases = [
        ("wing_yaw_forward", np.array([0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1.0], dtype=np.float32)),
        ("wing_yaw_backward", np.array([0, 0, 0, -0.8, 0, 0, -0.8, 0, 0, 0, 0, 1.0], dtype=np.float32)),
        ("wing_pitch_positive", np.array([0, 0, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 1.0], dtype=np.float32)),
        ("wing_pitch_negative", np.array([0, 0, 0, 0, 0, -0.8, 0, 0, -0.8, 0, 0, 1.0], dtype=np.float32)),
        ("abdomen_dorsal", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 1.0], dtype=np.float32)),
        ("abdomen_ventral", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 1.0], dtype=np.float32)),
        ("combo_yaw_abdomen_1", np.array([0, 0, 0, 0.9, 0, 0.5, 0.9, 0, 0.5, 0, 0.7, 1.0], dtype=np.float32)),
        ("combo_yaw_abdomen_2", np.array([0, 0, 0, -0.9, 0, -0.5, -0.9, 0, -0.5, 0, -1.0, 1.0], dtype=np.float32)),
    ]

    for name, action in test_cases:
        env.reset()
        physics = env.physics
        
        pitch_rates = []
        pitches = []
        
        for step in range(100):  # 100 steps * 0.0002s = 0.02s
            env.step(action)
            qpos = physics.data.qpos[:7]
            qvel = physics.data.qvel[:6]
            quat = qpos[3:7]
            euler = quat_to_euler(quat)
            ang_vel = qvel[3:6]
            
            pitches.append(euler[1])
            pitch_rates.append(ang_vel[1])
            
        print(f"[{name:<22}] Final Pitch: {np.rad2deg(pitches[-1]):>7.1f} deg | Max Pitch Rate: {np.rad2deg(max(pitch_rates, key=abs)):>8.1f} deg/s | Delta Pitch: {np.rad2deg(pitches[-1] - pitches[0]):>7.1f} deg")

if __name__ == "__main__":
    test_pitch_maneuvers()
