"""
3D Room Roaming Environment for BioFlight.
Places the Drosophila fruit fly inside a 3D enclosed room (walls + floor)
and generates continuous 3D wandering waypoints for free room exploration.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from flybody.fly_envs import flight_imitation
from src.data.flight_dataset import quat_to_euler


class RoomRoamingEnv:
    """
    3D Enclosed Room Roaming Environment for Fruit Fly locomotion.
    """

    def __init__(
        self,
        room_size: Tuple[float, float, float] = (10.0, 10.0, 5.0),  # x, y, z room bounds in cm
        action_scale: float = 3.0
    ):
        self.room_size = room_size
        self.action_scale = action_scale

        # Instantiate FlyBody environment without distance-based termination
        self.underlying_env = flight_imitation(terminal_com_dist=float('inf'))

        self.current_step = 0
        self.target_waypoint = self._generate_random_waypoint()

    @property
    def physics(self):
        return self.underlying_env.physics

    def _generate_random_waypoint(self) -> np.ndarray:
        """Generates a random 3D target waypoint inside room boundaries (in cm)."""
        rx, ry, rz = self.room_size
        x = np.random.uniform(-rx/2, rx/2)
        y = np.random.uniform(-ry/2, ry/2)
        z = np.random.uniform(0.5, rz)
        return np.array([x, y, z], dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.underlying_env.reset()
        self.current_step = 0
        self.target_waypoint = self._generate_random_waypoint()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Extracts 28D state observation relative to current room waypoint."""
        fly_qvel = self.physics.data.qvel[:6]
        lin_vel = fly_qvel[:3]
        ang_vel = fly_qvel[3:6]

        fly_qpos = self.physics.data.qpos[:7]
        pos = fly_qpos[:3]
        quat = fly_qpos[3:7]
        euler = quat_to_euler(quat)

        pos_err = self.target_waypoint - pos

        # Create 5 lookahead displacements toward room waypoint
        lookahead = []
        for k in range(1, 6):
            lookahead.extend(pos_err * (k / 5.0))

        obs = np.concatenate([
            lin_vel,
            ang_vel,
            quat,
            euler,
            np.array(lookahead, dtype=np.float32)
        ], dtype=np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        clipped_action = np.clip(action, -1.0, 1.0) * self.action_scale

        self.underlying_env.step(clipped_action)
        obs = self._get_obs()

        fly_pos = self.physics.data.qpos[:3]
        dist_to_waypoint = float(np.linalg.norm(self.target_waypoint - fly_pos))

        # Waypoint reached check: Pick new room waypoint when fly gets close (< 0.8 cm)
        waypoint_reached = dist_to_waypoint < 0.8
        if waypoint_reached:
            self.target_waypoint = self._generate_random_waypoint()

        crashed = fly_pos[2] < 0.1
        terminated = bool(crashed)
        truncated = False

        info = {
            "fly_pos_cm": fly_pos.copy(),
            "target_waypoint_cm": self.target_waypoint.copy(),
            "dist_to_waypoint_cm": dist_to_waypoint,
            "waypoint_reached": waypoint_reached,
            "crashed": crashed,
            "step": self.current_step
        }

        return obs, 1.0 - dist_to_waypoint, terminated, truncated, info
