"""
Spelling Flight Environment for FlyBody.

Wraps the official FlyBody simulation to execute sequential letter navigation:
- Instantiates 26 alphabet target spheres across the 3D flight arena ('A' through 'Z').
- Directs the fly controller to the current active letter target.
- Verifies physical contact ($d \le 0.15$ cm) in MuJoCo physics.
- Advances through the target word letters sequentially.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np

import flybody
from flybody import fly_envs
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from src.tasks.spelling import AlphabetArena, LetterTarget, SpellingTask
from src.data.flight_dataset import quat_to_euler


class SpellingFlightEnv:
    """
    Spelling task flight environment with 26 spatial letter targets.
    """

    def __init__(
        self,
        contact_radius: float = 0.15,
        max_steps_per_letter: int = 220,
        reward_letter_contact: float = 10.0,
        step_penalty: float = -0.01,
    ):
        self.alphabet = AlphabetArena(contact_radius=contact_radius)
        self.contact_radius = contact_radius
        self.max_steps = max_steps_per_letter
        self.reward_letter_contact = reward_letter_contact
        self.step_penalty = step_penalty

        # FlyBody environment with InferenceFlightTrajectoryLoader
        self.underlying_env = fly_envs.flight_imitation(
            ref_path=None,
            terminal_com_dist=float('inf')
        )
        self.dt = self.underlying_env.control_timestep()  # 0.0002 s (5000 Hz)

        self.active_letter: Optional[str] = None
        self.active_target: Optional[LetterTarget] = None
        self.step_count = 0
        self.min_dist_to_active = float('inf')

    @property
    def physics(self):
        return self.underlying_env.physics

    def reset_letter_leg(self, letter: str) -> Dict[str, Any]:
        """
        Initializes a flight leg towards the specified alphabet target letter.
        Generates a target-directed trajectory heading towards the letter's 3D position.
        """
        self.active_letter = letter.upper()
        self.active_target = self.alphabet.get_target(self.active_letter)
        self.step_count = 0
        self.min_dist_to_active = float('inf')

        # Heading angle toward the letter position from spawn (0, 0, 1.037)
        heading = float(np.arctan2(self.active_target.position[1], self.active_target.position[0]))

        # Target reference trajectory
        qpos, qvel = constant_speed_trajectory(
            n_steps=self.max_steps + 100,
            speed=20.0,
            init_pos=(0.0, 0.0, 1.037),
            init_heading=heading,
            body_rot_angle_y=-47.5,
            control_timestep=self.dt
        )
        self.underlying_env.task._traj_generator.set_next_trajectory(qpos, qvel)

        # Reset MuJoCo physics
        self.underlying_env.reset()

        return self.get_observation()

    def get_fly_position(self) -> np.ndarray:
        return self.physics.named.data.xpos['walker/thorax'].copy()

    def get_observation(self) -> Dict[str, Any]:
        physics = self.physics
        fly_qvel = physics.data.qvel[:6]
        fly_lin_vel = fly_qvel[:3].copy()
        fly_ang_vel = fly_qvel[3:6].copy()

        fly_qpos = physics.data.qpos[:7]
        fly_pos = fly_qpos[:3].copy()
        fly_quat = fly_qpos[3:7].copy()
        fly_euler = quat_to_euler(fly_quat)

        # 5-step lookahead displacements to target reference
        traj_gen = self.underlying_env.task._traj_generator
        com_qpos = getattr(traj_gen, '_com_qpos', None)
        cur_step = self.step_count

        future_disp = []
        for k in range(1, 6):
            target_step = cur_step + k
            if com_qpos is not None and isinstance(com_qpos, np.ndarray) and target_step < com_qpos.shape[0]:
                ref_point = com_qpos[target_step, :3]
            else:
                ref_point = self.active_target.position
            future_disp.extend(ref_point - fly_pos)

        raw_feature = np.concatenate([
            fly_lin_vel,
            fly_ang_vel,
            fly_quat,
            fly_euler,
            np.array(future_disp, dtype=np.float32)
        ]).astype(np.float32)

        dist_active = self.active_target.distance_to(fly_pos)

        return {
            "fly_pos": fly_pos,
            "fly_quat": fly_quat,
            "policy_input_feature": raw_feature,
            "active_letter": self.active_letter,
            "dist_to_active": dist_active
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.step_count += 1
        timestep = self.underlying_env.step(action)

        fly_pos = self.get_fly_position()
        dist_active = self.active_target.distance_to(fly_pos)
        self.min_dist_to_active = min(self.min_dist_to_active, dist_active)

        is_contact = self.active_target.is_contact(fly_pos)
        terminated = False
        success = False
        reward = self.step_penalty

        if is_contact:
            terminated = True
            success = True
            reward = self.reward_letter_contact
            reason = f"LETTER_{self.active_letter}_CONTACT"
        elif fly_pos[2] < 0.20:
            terminated = True
            success = False
            reward = -20.0
            reason = "GROUND_COLLISION"
        elif self.step_count >= self.max_steps or timestep.last():
            terminated = True
            success = False
            reward = -1.0
            reason = "TIMEOUT"
        else:
            reason = "NAVIGATING"

        obs = self.get_observation()
        info = {
            "step": self.step_count,
            "fly_pos": fly_pos.tolist(),
            "dist_to_active": float(dist_active),
            "min_dist_to_active": float(self.min_dist_to_active),
            "is_contact": is_contact,
            "success": success,
            "terminated": terminated,
            "termination_reason": reason,
            "reward": float(reward)
        }

        return obs, float(reward), terminated, info
