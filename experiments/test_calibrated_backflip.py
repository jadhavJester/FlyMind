"""
Calibrate exact single 360-degree loop stop and stable cruise recovery.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

def test_calibrated():
    env = flight_imitation(terminal_com_dist=float('inf'))
    dt = env.control_timestep()
    
    qp, qv = constant_speed_trajectory(
        n_steps=500,
        speed=15.0,
        init_pos=(0.0, 0.0, 1.037),
        body_rot_angle_y=-47.5,
        control_timestep=dt
    )
    env.task._traj_generator.set_next_trajectory(qp, qv)
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')

    # Forward cruise trim (purely symmetric)
    cruise_act = np.zeros(12, dtype=np.float32)
    cruise_act[3] = 0.1
    cruise_act[6] = 0.1
    cruise_act[11] = 0.4

    # Backflip loop pulse
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
    brake_act[3] = -0.7   # stroke backward hard to brake pitch
    brake_act[5] = -0.5
    brake_act[6] = -0.7
    brake_act[8] = -0.5
    brake_act[10] = -0.9  # abdomen ventral hard to push nose down
    brake_act[11] = 0.8   # good thrust to restore altitude

    # Test different loop durations
    for loop_steps in [85, 90, 95, 100]:
        env.reset()
        rot = 0.0
        records = []
        for step in range(200):
            if step < 15:
                act = cruise_act
            elif step < 15 + loop_steps:
                act = loop_act
            elif step < 15 + loop_steps + 40:
                act = brake_act
            else:
                act = cruise_act

            env.step(act)
            rot += env.physics.data.qvel[4] * dt
            pos = env.physics.data.xpos[thorax_id]
            xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
            records.append((step, np.rad2deg(rot), pos[2], xmat[0,0], xmat[2,0], xmat[2,2]))

        arr = np.array(records)
        max_rot = np.max(arr[:, 1])
        final_rot = arr[-1, 1]
        min_z = np.min(arr[:, 2])
        final_z = arr[-1, 2]
        final_fwd_x = arr[-1, 3]
        final_up_z = arr[-1, 5]
        print(f"Loop: {loop_steps:2d} steps | Peak: {max_rot:5.1f}° | Final: {final_rot:5.1f}° | Min Z: {min_z:.2f} cm | Final Z: {final_z:.2f} cm | Fwd_X: {final_fwd_x:.2f} | Up_Z: {final_up_z:.2f}")

if __name__ == "__main__":
    test_calibrated()
