"""
Test smooth action interpolation for realistic, non-jerky, perfectly consistent backflip.
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

def test_smooth():
    env = flight_imitation(terminal_com_dist=float('inf'))
    dt = env.control_timestep()
    
    # Cruising reference trajectory
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
    controller = FlightPolicyController()

    # Pre-climb entry action (gains upward momentum and altitude)
    climb_act = np.zeros(12, dtype=np.float32)
    climb_act[2] = -0.15  # head nod up slightly
    climb_act[3] = 0.35   # moderate forward sweep
    climb_act[5] = 0.40   # moderate AoA
    climb_act[6] = 0.35
    climb_act[8] = 0.40
    climb_act[10] = 0.30  # slight dorsal abdomen
    climb_act[11] = 1.0   # full thrust climb

    # Explosive pitch loop action
    loop_act = np.zeros(12, dtype=np.float32)
    loop_act[2] = -0.30  # head nod up
    loop_act[3] = 0.90   # forward stroke sweep
    loop_act[5] = 0.65   # high AoA
    loop_act[6] = 0.90
    loop_act[8] = 0.65
    loop_act[10] = 0.65  # dorsal abdomen
    loop_act[11] = 1.0   # full thrust

    # Pull-out / level brake action
    brake_act = np.zeros(12, dtype=np.float32)
    brake_act[2] = 0.15
    brake_act[3] = -0.35  # gentle backward sweep to arrest pitch
    brake_act[5] = -0.20
    brake_act[6] = -0.35
    brake_act[8] = -0.20
    brake_act[10] = -0.30
    brake_act[11] = 0.8   # strong thrust to climb out of dive

    current_act = np.zeros(12, dtype=np.float32)
    alpha = 0.15  # Smooth low-pass filter (exponential moving average)

    rot = 0.0
    records = []

    # Sequence:
    # 0..25: Cruise (5 ms)
    # 25..45: Climb entry (4 ms)
    # 45..155: Loop (22 ms)
    # 155..185: Brake & Pull-out (6 ms)
    # 185..260: Neural policy recovery (15 ms)
    total_steps = 260
    for step in range(total_steps):
        if step < 25:
            obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
            target_act = controller.policy.select_action(obs)
            phase = "1_CRUISE"
        elif step < 45:
            target_act = climb_act
            phase = "2_CLIMB_ENTRY"
        elif step < 155:
            target_act = loop_act
            phase = "3_PITCH_LOOP"
        elif step < 185:
            target_act = brake_act
            phase = "4_PULL_OUT"
        else:
            obs = extract_features(env, controller.mean, controller.std, lookahead_steps=5)
            target_act = controller.policy.select_action(obs)
            phase = "5_NEURAL_RECOVERY"

        # Apply smooth filter to eliminate actuator step shocks
        current_act = (1 - alpha) * current_act + alpha * target_act

        env.step(current_act)
        rot += env.physics.data.qvel[4] * dt
        pos = env.physics.data.xpos[thorax_id]
        xmat = env.physics.data.xmat[thorax_id].reshape(3, 3)
        records.append((step, phase, np.rad2deg(rot), pos[0], pos[2], xmat[0,0], xmat[2,0], xmat[2,2]))

    arr = np.array([(r[0], r[2], r[4]) for r in records])
    max_rot = np.max(arr[:, 1])
    min_z = np.min(arr[:, 2])
    final_z = arr[-1, 2]
    final_rot = arr[-1, 1]
    final_up_z = records[-1][7]
    final_fwd_x = records[-1][5]

    print(f"\nSmooth Backflip Results:")
    print(f"  Peak Pitch Rotation : {max_rot:6.1f}°")
    print(f"  Final Pitch Rotation: {final_rot:6.1f}°")
    print(f"  Min Altitude (Z)    : {min_z:6.2f} cm")
    print(f"  Final Altitude (Z)  : {final_z:6.2f} cm")
    print(f"  Final Up_Z          : {final_up_z:6.2f}")
    print(f"  Final Fwd_X         : {final_fwd_x:6.2f}")

    print("\nKey trajectory checkpoints:")
    for r in records[::20]:
        print(f"Step {r[0]:3d} [{r[1]:<17}]: Rot = {r[2]:6.1f}° | Pos = ({r[3]:.2f}, {r[4]:.2f}) cm | Fwd = ({r[5]:.2f}, {r[6]:.2f}) | Up_Z = {r[7]:.2f}")

if __name__ == "__main__":
    test_smooth()
