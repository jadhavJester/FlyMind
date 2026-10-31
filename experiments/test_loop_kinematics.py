"""
Detailed test of pitch loop progression and 360-degree backflip tracking.
"""

import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from src.data.flight_dataset import quat_to_euler


def test_loop_progression():
    env = flight_imitation(terminal_com_dist=float('inf'))
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')

    init_pos = env.physics.data.xpos[thorax_id].copy()
    init_xmat = env.physics.data.xmat[thorax_id].reshape(3, 3).copy()
    print(f"Thorax ID: {thorax_id}")
    print(f"Initial Thorax Pos: {init_pos}")
    print(f"Initial Forward vector: {init_xmat[:, 0]}")
    print(f"Initial Up vector: {init_xmat[:, 2]}")

    pitch_action = np.array([
        0.0,   # head_abduct
        0.0,   # head_twist
        -0.3,  # head (nod up)
        0.8,   # wing_yaw_left (forward sweep -> lift ahead of CoM)
        0.0,   # wing_roll_left
        0.6,   # wing_pitch_left (high AoA)
        0.8,   # wing_yaw_right (forward sweep)
        0.0,   # wing_roll_right
        0.6,   # wing_pitch_right (high AoA)
        0.0,   # abdomen_abduct
        0.6,   # abdomen (dorsal pitch trim)
        1.0,   # wbpg_frequency_bias (max lift)
    ], dtype=np.float32)

    total_pitch_rotation = 0.0
    prev_forward = init_xmat[:, 0].copy()

    history = []
    
    # 250 steps = 0.05 seconds
    for step in range(300):
        env.step(pitch_action)
        
        pos = env.physics.data.xpos[thorax_id].copy()
        xmat = env.physics.data.xmat[thorax_id].reshape(3, 3).copy()
        fwd = xmat[:, 0]
        lat = xmat[:, 1]
        up = xmat[:, 2]
        
        dot = np.clip(np.dot(prev_forward, fwd), -1.0, 1.0)
        d_angle = np.arccos(dot)
        
        cross_y = np.cross(prev_forward, fwd)
        sign = 1.0 if np.dot(cross_y, lat) > 0 else -1.0
        total_pitch_rotation += sign * d_angle
        prev_forward = fwd.copy()
        
        history.append((step, pos[0], pos[2], fwd[0], fwd[2], up[2], np.rad2deg(total_pitch_rotation)))

    print(f"\nFinal Total Pitch Rotation: {np.rad2deg(total_pitch_rotation):.1f} degrees")
    print(f"Sample trajectory points (step, x, z, fwd_x, fwd_z, up_z, total_deg):")
    for row in history[::25]:
        print(f"Step {row[0]:3d}: Pos=({row[1]:.2f}, {row[2]:.2f}) cm | Fwd=({row[3]:.2f}, {row[4]:.2f}) | Up_Z={row[5]:.2f} | PitchRot={row[6]:.1f}°")

if __name__ == "__main__":
    test_loop_progression()
