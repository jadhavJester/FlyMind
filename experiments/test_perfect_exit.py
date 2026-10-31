"""
Test precise 360-degree exit arrest and rock-solid level cruise.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

def test_perfect_exit():
    env = flight_imitation(terminal_com_dist=float('inf'))
    dt = env.control_timestep()
    
    qp, qv = constant_speed_trajectory(
        n_steps=600,
        speed=20.0,
        init_pos=(0.0, 0.0, 1.037),
        body_rot_angle_y=-47.5,
        control_timestep=dt
    )
    env.task._traj_generator.set_next_trajectory(qp, qv)
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')

    # 1. Cruise action
    cruise_act = np.zeros(12, dtype=np.float32)
    cruise_act[3] = 0.1
    cruise_act[6] = 0.1
    cruise_act[11] = 0.4

    # 2. Climb entry
    climb_act = np.zeros(12, dtype=np.float32)
    climb_act[2] = -0.15
    climb_act[3] = 0.35
    climb_act[5] = 0.40
    climb_act[6] = 0.35
    climb_act[8] = 0.40
    climb_act[10] = 0.30
    climb_act[11] = 1.0

    # 3. Dynamic Loop
    loop_act = np.zeros(12, dtype=np.float32)
    loop_act[2] = -0.30
    loop_act[3] = 0.90
    loop_act[5] = 0.65
    loop_act[6] = 0.90
    loop_act[8] = 0.65
    loop_act[10] = 0.65
    loop_act[11] = 1.0

    # 4. Pull-out brake
    brake_act = np.zeros(12, dtype=np.float32)
    brake_act[2] = 0.15
    brake_act[3] = -0.50
    brake_act[5] = -0.30
    brake_act[6] = -0.50
    brake_act[8] = -0.30
    brake_act[10] = -0.60
    brake_act[11] = 0.9

    # 5. Stable Level Cruise Recovery
    level_act = np.zeros(12, dtype=np.float32)
    level_act[3] = 0.05
    level_act[6] = 0.05
    level_act[10] = -0.05
    level_act[11] = 0.55

    current_act = np.zeros(12, dtype=np.float32)
    alpha = 0.15

    rot = 0.0
    records = []

    # Timing:
    # 0..20: Cruise
    # 20..40: Climb
    # 40..150: Loop
    # 150..185: Brake
    # 185..260: Level
    for step in range(250):
        if step < 20:
            target_act = cruise_act
            phase = "1_CRUISE"
        elif step < 40:
            target_act = climb_act
            phase = "2_CLIMB"
        elif step < 150:
            target_act = loop_act
            phase = "3_LOOP"
        elif step < 185:
            target_act = brake_act
            phase = "4_BRAKE"
        else:
            target_act = level_act
            phase = "5_LEVEL"

        current_act = (1 - alpha) * current_act + alpha * target_act

        env.step(current_act)
        rot += env.physics.data.qvel[4] * dt
        pos = env.physics.data.xpos[thorax_id]
        xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
        records.append((step, phase, np.rad2deg(rot), pos[0], pos[2], xmat[0,0], xmat[2,0], xmat[2,2]))

    for r in records[::15]:
        print(f"Step {r[0]:3d} [{r[1]:<7}]: Rot = {r[2]:6.1f}° | Pos = ({r[3]:.2f}, {r[4]:.2f}) cm | Fwd = ({r[5]:.2f}, {r[6]:.2f}) | Up_Z = {r[7]:.2f}")

    arr = np.array([(r[0], r[2], r[4]) for r in records])
    print(f"\nResult: Peak Rot = {np.max(arr[:, 1]):.1f}° | Min Z = {np.min(arr[:, 2]):.2f} cm | Final Z = {arr[-1, 2]:.2f} cm | Final Up_Z = {records[-1][7]:.2f} | Final Fwd_X = {records[-1][5]:.2f}")

if __name__ == "__main__":
    test_perfect_exit()
