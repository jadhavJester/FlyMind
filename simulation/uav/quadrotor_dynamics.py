"""
6-DoF Quadrotor Flight Dynamics Simulation for BioFlight (Phase 10).
Implements 6-DoF Newton-Euler quadrotor rotational and translational dynamics
with 4th-order Runge-Kutta (RK4) integration.
"""

import numpy as np
from typing import Tuple


class QuadrotorDynamics:
    """
    6-Degree-of-Freedom Quadrotor Rigidbody Dynamics.
    State vector (12D): [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    Control input (4D): [f1, f2, f3, f4] (Rotor thrust forces in Newtons)
    """

    def __init__(
        self,
        mass: float = 0.5,           # kg
        arm_length: float = 0.2,     # meters (center to motor)
        Ixx: float = 0.005,          # kg*m^2
        Iyy: float = 0.005,          # kg*m^2
        Izz: float = 0.009,          # kg*m^2
        drag_coeff: float = 0.05,    # Linear aerodynamic drag
        km: float = 0.01             # Motor torque constant (tau_z = km * thrust)
    ):
        self.m = mass
        self.g = 9.81
        self.l = arm_length
        self.I = np.diag([Ixx, Iyy, Izz])
        self.I_inv = np.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz])
        self.drag_coeff = drag_coeff
        self.km = km

        # Hover thrust per motor
        self.hover_thrust = (self.m * self.g) / 4.0

    def rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Computes ZYX rotation matrix from Euler angles."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr]
        ], dtype=np.float32)
        return R

    def derivatives(self, state: np.ndarray, rotor_thrusts: np.ndarray, wind_gust: np.ndarray = None) -> np.ndarray:
        """
        Computes 12D state derivatives dX/dt.
        """
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]   # [roll, pitch, yaw]
        ang_vel = state[9:12] # [wx, wy, wz]

        f1, f2, f3, f4 = rotor_thrusts
        total_thrust = f1 + f2 + f3 + f4

        # Torques in body frame (X configuration quadrotor)
        tau_x = (self.l / np.sqrt(2)) * (f1 + f2 - f3 - f4)
        tau_y = (self.l / np.sqrt(2)) * (-f1 + f2 + f3 - f4)
        tau_z = self.km * (-f1 + f2 - f3 + f4)
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        # Translational acceleration
        R = self.rotation_matrix(angles[0], angles[1], angles[2])
        thrust_body = np.array([0.0, 0.0, total_thrust], dtype=np.float32)
        gravity_world = np.array([0.0, 0.0, -self.m * self.g], dtype=np.float32)
        
        drag_force = -self.drag_coeff * vel
        if wind_gust is not None:
            drag_force += -self.drag_coeff * (vel - wind_gust)

        accel = (R @ thrust_body + gravity_world + drag_force) / self.m

        # Rotational acceleration
        gyroscopic = np.cross(ang_vel, self.I @ ang_vel)
        ang_accel = self.I_inv @ (torques - gyroscopic)

        # Euler angle rates (small angle approximation for simplicity and stability)
        euler_rates = ang_vel.copy()

        d_state = np.concatenate([vel, accel, euler_rates, ang_accel], dtype=np.float32)
        return d_state

    def step_rk4(self, state: np.ndarray, rotor_thrusts: np.ndarray, dt: float = 0.001, wind_gust: np.ndarray = None) -> np.ndarray:
        """
        4th-Order Runge-Kutta numerical integration.
        """
        k1 = self.derivatives(state, rotor_thrusts, wind_gust)
        k2 = self.derivatives(state + 0.5 * dt * k1, rotor_thrusts, wind_gust)
        k3 = self.derivatives(state + 0.5 * dt * k2, rotor_thrusts, wind_gust)
        k4 = self.derivatives(state + dt * k3, rotor_thrusts, wind_gust)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state
