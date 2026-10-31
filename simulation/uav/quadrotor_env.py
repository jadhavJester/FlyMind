"""
Quadrotor Environment for BioFlight (Phase 10).
Provides Gym-style simulation interface for Quadrotor flight dynamics and tracking tasks.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from simulation.uav.quadrotor_dynamics import QuadrotorDynamics


class QuadrotorEnv:
    """
    Simulation Environment for Quadrotor UAV Dynamics.
    """

    def __init__(
        self,
        dt: float = 0.001,
        max_steps: int = 500,
        wind_std: float = 0.0
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.wind_std = wind_std
        self.dynamics = QuadrotorDynamics()

        # State (12D): [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state = np.zeros(12, dtype=np.float32)
        self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.current_step = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.state = np.zeros(12, dtype=np.float32)
        self.state[2] = 1.0  # Initial height 1.0 m
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, rotor_thrusts: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        wind_gust = np.random.normal(0.0, self.wind_std, size=3).astype(np.float32) if self.wind_std > 0 else None
        self.state = self.dynamics.step_rk4(self.state, rotor_thrusts, dt=self.dt, wind_gust=wind_gust)

        pos = self.state[:3]
        pos_err = np.linalg.norm(self.target_pos - pos)

        reward = 1.0 - pos_err

        crashed = pos[2] < 0.1 or abs(self.state[6]) > 1.2 or abs(self.state[7]) > 1.2
        terminated = bool(crashed)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "pos_err_m": float(pos_err),
            "crashed": crashed,
            "step": self.current_step
        }

        return self.state.copy(), float(reward), terminated, truncated, info
