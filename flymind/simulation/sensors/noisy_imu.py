"""
Noisy IMU Sensor Model.

Simulates a 6-axis Inertial Measurement Unit (3-axis gyroscope and 3-axis accelerometer)
with realistic additive Gaussian white noise and slowly drifting random-walk bias.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple


class NoisyIMU:
    """
    Simulated IMU mounted at drone Center of Mass (CoM).
    Generates gyro angular rate (rad/s) and accelerometer specific force (m/s^2).
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        dt: float | None = None,
        seed: int = 42,
    ):
        cfg = (config or {}).get("imu", {})
        self.gyro_noise_density = float(cfg.get("gyro_noise_density", 0.001))
        self.gyro_random_walk = float(cfg.get("gyro_random_walk", 0.0001))
        self.accel_noise_density = float(cfg.get("accel_noise_density", 0.01))
        self.accel_random_walk = float(cfg.get("accel_random_walk", 0.001))
        self.update_hz = float(cfg.get("update_hz", 100.0))
        self.dt = dt if dt is not None else (1.0 / self.update_hz)

        self.rng = np.random.default_rng(seed)
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)

    def reset(self):
        """Resets the random walk biases to zero."""
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)

    def measure(
        self,
        true_angular_vel: np.ndarray,
        true_linear_accel: np.ndarray,
        drone_euler: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience measurement interface."""
        euler = drone_euler if drone_euler is not None else np.zeros(3, dtype=np.float64)
        return self.update(true_angular_vel, true_linear_accel, euler)

    def update(
        self,
        true_angular_vel: np.ndarray,
        true_linear_accel_world: np.ndarray,

        drone_euler: np.ndarray,
        dt: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes true body angular rates and world linear acceleration, and returns
        noisy measurements (gyro_meas [rad/s], accel_meas [m/s^2]).
        """
        step_dt = dt if dt is not None else self.dt
        sq_dt = np.sqrt(step_dt)

        # Update random-walk biases
        self.gyro_bias += self.gyro_random_walk * sq_dt * self.rng.standard_normal(3)
        self.accel_bias += self.accel_random_walk * sq_dt * self.rng.standard_normal(3)

        # White measurement noise
        gyro_noise = (self.gyro_noise_density / sq_dt) * self.rng.standard_normal(3)
        accel_noise = (self.accel_noise_density / sq_dt) * self.rng.standard_normal(3)

        # Gyro reading: body angular velocity + bias + noise
        gyro_meas = true_angular_vel + self.gyro_bias + gyro_noise

        # Accelerometer reading: specific force = R^T * (accel_world - gravity_world)
        roll, pitch, yaw = drone_euler
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr]
        ], dtype=np.float64)

        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        specific_force_world = true_linear_accel_world - gravity_world
        specific_force_body = R.T @ specific_force_world

        accel_meas = specific_force_body + self.accel_bias + accel_noise
        return gyro_meas, accel_meas
