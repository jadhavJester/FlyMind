"""
Test symmetric backflip with forward run-up, loop, brake, and recovery.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

def test_symmetric():
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
    cruise_act[3] = 0.1   # slight forward stroke
    cruise_act[6] = 0.1
    cruise_act[11] = 0.4  # cruising thrust

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
    brake_act[3] = -0.6
    brake_act[5] = -0.4
    brake_act[6] = -0.6
    brake_act[8] = -0.4
    brake_act[10] = -0.8
    brake_act[11] = 0.6

    # Recovery cruise
    recov_act = np.zeros(12, dtype=np.float32)
    recov_act[3] = 0.1
    recov_act[6] = 0.1
    recov_act[11] = 0.5

    rot = 0.0
    records = []

    # Cruise: 20 steps (4 ms)
    # Loop: 112 steps (22.4 ms)
    # Brake: 25 steps (5 ms)
    # Recover: 60 steps (12 ms)
    for step in range(220):
        if step < 20:
            act = cruise_act
            phase = "CRUISE"
        elif step < 132:
            act = loop_act
            phase = "LOOP"
        elif step < 157:
            act = brake_act
            phase = "BRAKE"
        else:
            act = recov_act
            phase = "RECOVER"

        env.step(act)
        rot += env.physics.data.qvel[4] * dt
        pos = env.physics.data.xpos[thorax_id]
        xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
        records.append((step, np.rad2deg(rot), pos[0], pos[2], xmat[0,0], xmat[2,0], xmat[2,2], phase))

    for r in records[::15]:
        print(f"Step {r[0]:3d} [{r[7]:<7}]: Rot = {r[1]:6.1f}° | Pos = ({r[2]:.2f}, {r[3]:.2f}) cm | Fwd = ({r[4]:.2f}, {r[5]:.2f}) | Up_Z = {r[6]:.2f}")

    arr = np.array([(r[0], r[1], r[2], r[3]) for r in records])
    max_rot = np.max(arr[:, 1])
    min_z = np.min(arr[:, 3])
    final_z = arr[-1, 3]
    final_rot = arr[-1, 1]
    print(f"\nSummary: Max Rot = {max_rot:.1f}° | Final Rot = {final_rot:.1f}° | Min Z = {min_z:.2f} cm | Final Z = {final_z:.2f} cm")

if __name__ == "__main__":
    test_symmetric()
