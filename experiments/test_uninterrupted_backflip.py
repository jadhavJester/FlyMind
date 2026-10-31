"""
Test uninterrupted 360-degree backflip with 500-step trajectory.
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

def test():
    env = flight_imitation(terminal_com_dist=float('inf'))
    dt = env.control_timestep()
    
    # 500-step trajectory so it never terminates early
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
    controller = FlightPolicyController()

    loop_act = np.zeros(12, dtype=np.float32)
    loop_act[2] = -0.3   # head nod up
    loop_act[3] = 0.9    # wing_yaw_left forward sweep
    loop_act[5] = 0.7    # wing_pitch_left high AoA
    loop_act[6] = 0.9    # wing_yaw_right forward sweep
    loop_act[8] = 0.7    # wing_pitch_right high AoA
    loop_act[10] = 0.7   # abdomen dorsal flexion
    loop_act[11] = 1.0   # frequency boost

    rot = 0.0
    for step in range(160):
        if step < 25:
            obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
            act = controller.policy.select_action(obs)
        elif step < 140:
            act = loop_act
        else:
            obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
            act = controller.policy.select_action(obs)

        env.step(act)
        rot += env.physics.data.qvel[4] * dt
        xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
        pos = env.physics.data.xpos[thorax_id]
        if step % 10 == 0:
            print(f"Step {step:3d}: Rot = {np.rad2deg(rot):6.1f}° | Pos = ({pos[0]:.2f}, {pos[2]:.2f}) cm | Fwd = ({xmat[0,0]:.2f}, {xmat[2,0]:.2f}) | Up = ({xmat[0,2]:.2f}, {xmat[2,2]:.2f})")

if __name__ == "__main__":
    test()
