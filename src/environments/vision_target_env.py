"""
Vision-Guided Target Environment for BioFlight (Phase 9).
Provides visual observations (64x64 RGB camera image) of a target sphere
positioned at variable spatial offsets ahead of the fruit fly.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from flybody.fly_envs import flight_imitation
from src.data.flight_dataset import quat_to_euler


class VisionTargetEnv:
    """
    Vision-guided flight task environment rendering visual target observations.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
        max_steps: int = 300,
        action_scale: float = 3.0
    ):
        self.image_shape = image_shape
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.current_step = 0

        # Instantiate FlyBody environment
        self.underlying_env = flight_imitation()
        self.target_pos = np.array([2.0, 0.0, 1.0], dtype=np.float32)  # Target 2cm ahead

    @property
    def physics(self):
        return self.underlying_env.physics

    def _render_vision_frame(self) -> np.ndarray:
        """
        Renders 64x64 visual camera observation frame depicting target relative position.
        """
        fly_pos = self.physics.data.qpos[:3]  # cm
        rel_target = self.target_pos - fly_pos

        # Generate synthetic visual camera frame (64x64 RGB) for high-speed local processing
        img = np.zeros(self.image_shape, dtype=np.uint8)

        # Map target relative angle to image coordinates
        azimuth = np.arctan2(rel_target[1], rel_target[0] + 1e-6)
        elevation = np.arctan2(rel_target[2], rel_target[0] + 1e-6)

        cx = int(32 + azimuth * 20)
        cy = int(32 - elevation * 20)

        cx = np.clip(cx, 4, 59)
        cy = np.clip(cy, 4, 59)

        # Draw red visual target blob
        img[max(0, cy-3):min(64, cy+4), max(0, cx-3):min(64, cx+4), 0] = 255
        return img

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Returns visual image frame and kinematic state observation."""
        image = self._render_vision_frame()
        
        fly_qvel = self.physics.data.qvel[:6]
        lin_vel = fly_qvel[:3]
        ang_vel = fly_qvel[3:6]

        fly_qpos = self.physics.data.qpos[:7]
        pos = fly_qpos[:3]
        quat = fly_qpos[3:7]
        euler = quat_to_euler(quat)
        rel_pos = self.target_pos - pos

        kinematic_state = np.concatenate([lin_vel, ang_vel, quat, euler, rel_pos], dtype=np.float32)

        return {
            "image": image,
            "kinematic": kinematic_state
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            # Randomize target offset: Left / Right / Up / Down
            y_offset = np.random.uniform(-1.5, 1.5)
            z_offset = np.random.uniform(0.5, 2.0)
            self.target_pos = np.array([2.5, y_offset, z_offset], dtype=np.float32)

        self.underlying_env.reset()
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        clipped_action = np.clip(action, -1.0, 1.0) * self.action_scale

        self.underlying_env.step(clipped_action)
        obs = self._get_obs()

        fly_pos = self.physics.data.qpos[:3]
        dist_to_target = float(np.linalg.norm(self.target_pos - fly_pos))

        # Reward: +1 for survival, minus distance to target
        reward = 1.0 - dist_to_target

        crashed = fly_pos[2] < 0.1
        terminated = bool(crashed or dist_to_target < 0.3)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "dist_to_target_cm": dist_to_target,
            "target_reached": dist_to_target < 0.3,
            "crashed": crashed
        }

        return obs, reward, terminated, truncated, info
