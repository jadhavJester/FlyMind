"""
Experiment: Consistent, Stable Drosophila 360-Degree Backflip with Neural Recovery.
Tests clean loop completion and closed-loop neural stabilization after recovery.
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from src.agents.oracle_selector import FlightPolicyController
from evaluation.run_evaluation import extract_features
from src.data.flight_dataset import quat_to_euler

def test_consistent():
    env = flight_imitation(terminal_com_dist=float('inf'))
    dt = env.control_timestep()
    
    # Generate long straight cruising trajectory at z = 1.037 cm, speed = 20 cm/s
    speed = 20.0
    qp, qv = constant_speed_trajectory(
        n_steps=600,
        speed=speed,
        init_pos=(0.0, 0.0, 1.037),
        body_rot_angle_y=-47.5,
        control_timestep=dt
    )
    env.task._traj_generator.set_next_trajectory(qp, qv)
    env.reset()
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')
    controller = FlightPolicyController()

    # Backflip pulse action
    loop_act = np.zeros(12, dtype=np.float32)
    loop_act[2] = -0.3   # head nod up
    loop_act[3] = 0.95   # wing_yaw_left forward sweep
    loop_act[5] = 0.70   # wing_pitch_left high AoA
    loop_act[6] = 0.95   # wing_yaw_right forward sweep
    loop_act[8] = 0.70   # wing_pitch_right high AoA
    loop_act[10] = 0.70  # abdomen dorsal flexion
    loop_act[11] = 1.0   # frequency boost

    # Arrest brake action (smooth dampening)
    brake_act = np.zeros(12, dtype=np.float32)
    brake_act[2] = 0.1
    brake_act[3] = -0.4
    brake_act[5] = -0.2
    brake_act[6] = -0.4
    brake_act[8] = -0.2
    brake_act[10] = -0.4
    brake_act[11] = 0.6

    for loop_duration in [110, 115, 120, 125]:
        env.reset()
        init_xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
        init_pitch = np.arctan2(init_xmat[2, 0], init_xmat[0, 0])
        
        rot_deg = 0.0
        records = []
        raw_angles = []

        # 30 steps cruise, loop_duration steps loop, 20 steps brake, 80 steps neural recovery
        cruise_steps = 30
        brake_steps = 20
        total_steps = 220

        for step in range(total_steps):
            xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
            cur_angle = np.arctan2(xmat[2, 0], xmat[0, 0])
            raw_angles.append(cur_angle)
            unwrapped = np.unwrap(raw_angles)
            cur_rot = float(np.rad2deg(np.abs(unwrapped[-1] - unwrapped[0])))

            if step < cruise_steps:
                phase = "1_CRUISE"
                obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
                act = controller.policy.select_action(obs)
            elif step < cruise_steps + loop_duration:
                phase = "2_PITCH_LOOP"
                act = loop_act
            elif step < cruise_steps + loop_duration + brake_steps:
                phase = "3_PITCH_BRAKE"
                act = brake_act
            else:
                phase = "4_NEURAL_RECOVERY"
                obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
                act = controller.policy.select_action(obs)

            env.step(act)
            pos = env.physics.data.xpos[thorax_id]
            qvel = env.physics.data.qvel[:6]
            records.append((step, phase, cur_rot, pos[0], pos[1], pos[2], xmat[0,0], xmat[2,0], xmat[2,2]))

        arr_rot = [r[2] for r in records]
        arr_z = [r[5] for r in records]
        final_up = records[-1][8]
        final_fwd = records[-1][6]
        print(f"Duration {loop_duration:3d}: Peak Rot = {max(arr_rot):5.1f}° | Final Rot = {arr_rot[-1]:5.1f}° | Min Z = {min(arr_z):.2f} cm | Final Z = {arr_z[-1]:.2f} cm | Final Up_Z = {final_up:.2f} | Final Fwd_X = {final_fwd:.2f}")

if __name__ == "__main__":
    test_consistent()
