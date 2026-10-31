"""
Measure exact backflip steps and rotation profile.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation

def measure():
    env = flight_imitation(terminal_com_dist=float('inf'))
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')
    
    act = np.zeros(12, dtype=np.float32)
    act[2] = -0.3   # head nod up
    act[3] = 0.9    # wing_yaw_left forward sweep
    act[5] = 0.7    # wing_pitch_left high AoA
    act[6] = 0.9    # wing_yaw_right forward sweep
    act[8] = 0.7    # wing_pitch_right high AoA
    act[10] = 0.7   # abdomen dorsal flexion
    act[11] = 1.0   # frequency boost

    rot = 0.0
    dt = env.control_timestep()
    
    for step in range(250):
        env.step(act)
        rot += env.physics.data.qvel[4] * dt
        z = env.physics.data.xpos[thorax_id, 2]
        x = env.physics.data.xpos[thorax_id, 0]
        if step % 15 == 0:
            print(f"Step {step:3d}: Rotation = {np.rad2deg(rot):6.1f}° | Pos = ({x:.2f}, {z:.2f}) cm | Pitch Rate = {np.rad2deg(env.physics.data.qvel[4]):6.1f}°/s")

if __name__ == "__main__":
    measure()
