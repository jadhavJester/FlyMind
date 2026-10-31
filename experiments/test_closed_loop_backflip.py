"""
Closed-loop state-triggered 360-degree backflip.
Triggers each aerodynamic phase based on the fly's real-time pitch rotation angle.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

def test_closed_loop_backflip():
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

    # Forward cruise trim
    cruise_act = np.zeros(12, dtype=np.float32)
    cruise_act[3] = 0.1
    cruise_act[6] = 0.1
    cruise_act[11] = 0.5

    # High-performance pitch loop pulse
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
    brake_act[3] = -0.7   # stroke backward
    brake_act[5] = -0.4
    brake_act[6] = -0.7
    brake_act[8] = -0.4
    brake_act[10] = -0.8  # abdomen ventral
    brake_act[11] = 0.8   # thrust boost to hold altitude

    rot = 0.0
    phase = "ENTRY"
    brake_counter = 0

    records = []
    for step in range(220):
        # State machine
        if phase == "ENTRY":
            act = cruise_act
            if step >= 15:
                phase = "LOOP"
        elif phase == "LOOP":
            act = loop_act
            # Trigger brake when rotation approaches completion (between 300 and 320 degrees)
            if np.rad2deg(rot) >= 300.0:
                phase = "BRAKE"
                brake_counter = 0
        elif phase == "BRAKE":
            act = brake_act
            brake_counter += 1
            # Once rotation reaches ~360 deg or brake applied for 25 steps, transition to level flight
            if np.rad2deg(rot) >= 360.0 or brake_counter >= 25:
                phase = "CRUISE"
        elif phase == "CRUISE":
            act = cruise_act

        env.step(act)
        rot += env.physics.data.qvel[4] * dt
        pos = env.physics.data.xpos[thorax_id]
        xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
        qvel = env.physics.data.qvel[:6]
        records.append((step, np.rad2deg(rot), pos[0], pos[2], xmat[0,0], xmat[2,0], xmat[2,2], np.rad2deg(qvel[4]), phase))

    for r in records[::15]:
        print(f"Step {r[0]:3d} [{r[8]:<6}]: Rot = {r[1]:6.1f}° | Pos = ({r[2]:.2f}, {r[3]:.2f}) cm | Fwd = ({r[4]:.2f}, {r[5]:.2f}) | Up_Z = {r[6]:.2f} | PitchRate = {r[7]:6.1f}°/s")

    arr = np.array([(r[0], r[1], r[3]) for r in records])
    max_rot = np.max(arr[:, 1])
    min_z = np.min(arr[:, 2])
    final_z = arr[-1, 2]
    final_rot = arr[-1, 1]
    print(f"\nResult: Peak Rot = {max_rot:.1f}° | Final Rot = {final_rot:.1f}° | Min Z = {min_z:.2f} cm | Final Z = {final_z:.2f} cm")

if __name__ == "__main__":
    test_closed_loop_backflip()
