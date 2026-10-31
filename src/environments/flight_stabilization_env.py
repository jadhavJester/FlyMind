"""
Flight Stabilization RL Environment for BioFlight (Phase 8).
Wraps FlyBody MuJoCo physics into a clean Gym-style environment for RL training.
Target: Maintain stable hover/flight at target position (0, 0, 1.0 cm) and zero angular rates.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from flybody.fly_envs import flight_imitation
from src.data.flight_dataset import quat_to_euler


class FlightStabilizationEnv:
    """
    Custom RL Environment for Drosophila flight stabilization and hover control.
    """

    def __init__(
        self,
        target_pos: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),  # cm
        target_vel: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),  # cm/s
        max_steps: int = 500,
        action_scale: float = 3.0,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        self.target_pos = target_pos
        self.target_vel = target_vel
        self.max_steps = max_steps
        self.action_scale = action_scale
        
        # Default reward weights
        self.weights = reward_weights or {
            "alive": 1.0,
            "pos": 0.5,
            "vel": 0.1,
            "orient": 0.2,
            "effort": 0.01
        }

        # Instantiate underlying FlyBody physics env
        self.underlying_env = flight_imitation()

        self.observation_dim = 16
        self.action_dim = 12
        self.current_step = 0

    @property
    def physics(self):
        return self.underlying_env.physics

    def _get_obs(self) -> np.ndarray:
        """Extracts 16D state observation from MuJoCo physics."""
        fly_qvel = self.physics.data.qvel[:6]  # [vx, vy, vz, wx, wy, wz] (cm/s, rad/s)
        lin_vel = fly_qvel[:3]
        ang_vel = fly_qvel[3:6]

        fly_qpos = self.physics.data.qpos[:7]  # [x, y, z, qw, qx, qy, qz]
        pos = fly_qpos[:3]  # cm
        quat = fly_qpos[3:7]
        euler = quat_to_euler(quat)  # [roll, pitch, yaw] in rad

        pos_err = self.target_pos - pos

        obs = np.concatenate([
            lin_vel,
            ang_vel,
            quat,
            euler,
            pos_err
        ], dtype=np.float32)
        return obs

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets simulation environment."""
        if seed is not None:
            np.random.seed(seed)

        self.underlying_env.reset()
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applies continuous 12D wing action vector and steps MuJoCo physics.
        Returns (obs, reward, terminated, truncated, info).
        """
        self.current_step += 1
        
        # Clip & scale action
        clipped_action = np.clip(action, -1.0, 1.0) * self.action_scale

        # Step underlying physics
        self.underlying_env.step(clipped_action)

        obs = self._get_obs()

        # Extract state components for reward calculation
        lin_vel = obs[:3]
        ang_vel = obs[3:6]
        euler = obs[10:13]  # [roll, pitch, yaw]
        pos_err = obs[13:16]

        # Calculate penalty components
        pos_penalty = np.sum(np.square(pos_err))
        vel_penalty = np.sum(np.square(lin_vel - self.target_vel))
        orient_penalty = np.square(euler[0]) + np.square(euler[1])  # Roll & Pitch deviation
        effort_penalty = np.mean(np.square(action))

        # Reward
        reward = (
            self.weights["alive"]
            - self.weights["pos"] * pos_penalty
            - self.weights["vel"] * vel_penalty
            - self.weights["orient"] * orient_penalty
            - self.weights["effort"] * effort_penalty
        )

        # Check termination / crash conditions
        fly_pos = self.physics.data.qpos[:3]
        fly_height = fly_pos[2]  # z position
        roll_deg = np.degrees(abs(euler[0]))
        pitch_deg = np.degrees(abs(euler[1]))

        crashed = (fly_height < 0.1) or (roll_deg > 120.0) or (pitch_deg > 120.0)
        terminated = bool(crashed)
        truncated = bool(self.current_step >= self.max_steps)

        if terminated:
            reward -= 10.0  # Crash penalty

        info = {
            "pos_err_cm": float(np.linalg.norm(pos_err)),
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "crashed": crashed,
            "step": self.current_step
        }

        return obs, float(reward), terminated, truncated, info
