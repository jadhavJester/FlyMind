"""
6-DoF Quadrotor Flight Dynamics Simulation.

Implements full rigid-body Newton-Euler translational and rotational equations of motion
integrated via 4th-order Runge-Kutta (RK4). Includes rotor aerodynamics, linear and quadratic
bluff-body drag, and a cascaded attitude stabilization loop.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple


class Quadrotor6DoF:
    """
    6-Degree-of-Freedom Quadrotor Dynamic Model.

    State Vector (12 dims):
      [0:3]  Position in world frame [x, y, z] (meters, z-up)
      [3:6]  Linear velocity in world frame [vx, vy, vz] (m/s)
      [6:9]  Euler angles [roll, pitch, yaw] (radians, ZYX convention)
      [9:12] Body angular rates [p, q, r] (rad/s)
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        dt: float = 0.02,
        initial_pos: np.ndarray | None = None,
    ):
        self.default_dt = dt
        cfg = (config or {}).get("quadrotor", {})
        self.mass = float(cfg.get("mass_kg", 0.85))
        self.arm_length = float(cfg.get("arm_length_m", 0.17))
        self.gravity = float(cfg.get("gravity_mps2", 9.81))
        self.Ixx = float(cfg.get("Ixx_kgm2", 0.0075))
        self.Iyy = float(cfg.get("Iyy_kgm2", 0.0075))
        self.Izz = float(cfg.get("Izz_kgm2", 0.0130))
        self.linear_drag = float(cfg.get("linear_drag_coeff", 0.08))
        self.angular_drag = float(cfg.get("angular_drag_coeff", 0.02))
        self.km = float(cfg.get("motor_torque_constant", 0.012))
        self.max_motor_thrust = float(cfg.get("max_thrust_per_motor_n", 4.5))
        self.min_motor_thrust = float(cfg.get("min_thrust_per_motor_n", 0.1))

        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.I_inv = np.diag([1.0 / self.Ixx, 1.0 / self.Iyy, 1.0 / self.Izz])
        self.hover_thrust = (self.mass * self.gravity) / 4.0

        # Current state [12D]
        self.state = np.zeros(12, dtype=np.float64)
        if initial_pos is not None:
            self.state[0:3] = np.array(initial_pos, dtype=np.float64)
        else:
            self.state[2] = 1.0  # Initial hover altitude: 1.0 m

        # Internal controller PID gains for attitude loop
        self.kp_ang = np.array([4.5, 4.5, 2.5], dtype=np.float64)
        self.kd_ang = np.array([0.6, 0.6, 0.4], dtype=np.float64)
        self.kp_vel = np.array([1.2, 1.2, 2.0], dtype=np.float64)
        self.kd_vel = np.array([0.4, 0.4, 0.5], dtype=np.float64)

    @property
    def pos(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def yaw(self) -> float:
        return float(self.state[8])

    @property
    def omega(self) -> np.ndarray:
        return self.state[9:12]

    @property
    def rot_mat(self) -> np.ndarray:
        return self.rotation_matrix(self.state[6], self.state[7], self.state[8])

    def step_velocity_body(
        self,
        vx_body: float,
        vy_body: float,
        vz_body: float,
        yaw_rate_cmd: float,
        dt: float | None = None,
    ) -> np.ndarray:
        """Step simulation using body-frame velocity setpoint."""
        step_dt = dt if dt is not None else self.default_dt
        yaw = self.state[8]
        # Rotate body-frame velocity command to world frame
        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_w = cy * vx_body - sy * vy_body
        vy_w = sy * vx_body + cy * vy_body
        vz_w = -vz_body  # NED conversion: -vz_body -> positive upward in world frame
        v_world = np.array([vx_w, vy_w, vz_w], dtype=np.float64)

        thrusts = self.setpoint_to_motor_thrusts(v_world, yaw_rate_cmd)
        return self.step(thrusts, step_dt)

    def step_rk4(self, dt: float, motor_thrusts: np.ndarray, wind_vector: np.ndarray | None = None) -> np.ndarray:
        """Alias for step with (dt, motor_thrusts) argument order."""
        return self.step(motor_thrusts, dt, wind_vector)

    def reset(self, position: Tuple[float, float, float] = (0.0, 0.0, 1.0), yaw_deg: float = 0.0) -> np.ndarray:


        """Resets the drone state to rest at given position and heading."""
        self.state = np.zeros(12, dtype=np.float64)
        self.state[0:3] = np.array(position, dtype=np.float64)
        self.state[8] = np.deg2rad(yaw_deg)
        return self.state.copy()

    def rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Computes ZYX rotation matrix from body to world coordinates."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr]
        ], dtype=np.float64)
        return R

    def compute_derivatives(self, state: np.ndarray, motor_thrusts: np.ndarray, wind_vector: np.ndarray | None = None) -> np.ndarray:
        """Computes instantaneous state derivatives dX/dt."""
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]   # [roll, pitch, yaw]
        rates = state[9:12]   # [p, q, r]

        roll, pitch, yaw = angles[0], angles[1], angles[2]
        R = self.rotation_matrix(roll, pitch, yaw)

        # Total vertical thrust vector in body frame
        f_total = np.sum(motor_thrusts)
        thrust_body = np.array([0.0, 0.0, f_total], dtype=np.float64)
        thrust_world = R @ thrust_body

        # Gravity and aerodynamic drag
        gravity_world = np.array([0.0, 0.0, -self.mass * self.gravity], dtype=np.float64)
        rel_vel = vel if wind_vector is None else vel - wind_vector
        drag_force = -self.linear_drag * rel_vel * np.linalg.norm(rel_vel + 1e-6)

        # Translational acceleration
        accel = (thrust_world + gravity_world + drag_force) / self.mass

        # Motor torques in body frame (standard cross configuration)
        # f1: Front, f2: Right, f3: Rear, f4: Left
        tau_x = self.arm_length * (motor_thrusts[3] - motor_thrusts[1])  # Roll
        tau_y = self.arm_length * (motor_thrusts[0] - motor_thrusts[2])  # Pitch
        tau_z = self.km * (motor_thrusts[0] - motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[3]) # Yaw
        tau = np.array([tau_x, tau_y, tau_z], dtype=np.float64)
        tau_drag = -self.angular_drag * rates * np.abs(rates)

        # Rotational dynamics: I * d_omega/dt = tau - omega x (I * omega)
        omega_cross_I_omega = np.cross(rates, self.I @ rates)
        alpha = self.I_inv @ (tau + tau_drag - omega_cross_I_omega)

        # Euler angle rate kinematics
        cr, sr = np.cos(roll), np.sin(roll)
        tp = np.tan(pitch)
        cp = np.cos(pitch)
        cp = np.sign(cp) * max(abs(cp), 1e-4)

        W_inv = np.array([
            [1.0, sr * tp, cr * tp],
            [0.0, cr,      -sr],
            [0.0, sr / cp, cr / cp]
        ], dtype=np.float64)
        angle_rates = W_inv @ rates

        d_state = np.zeros(12, dtype=np.float64)
        d_state[0:3] = vel
        d_state[3:6] = accel
        d_state[6:9] = angle_rates
        d_state[9:12] = alpha
        return d_state

    def step(self, motor_thrusts: np.ndarray, dt: float = 0.01, wind_vector: np.ndarray | None = None) -> np.ndarray:
        """Integrates 6-DoF dynamics forward using 4th-order Runge-Kutta (RK4)."""
        thrusts = np.clip(motor_thrusts, self.min_motor_thrust, self.max_motor_thrust)

        k1 = self.compute_derivatives(self.state, thrusts, wind_vector)
        k2 = self.compute_derivatives(self.state + 0.5 * dt * k1, thrusts, wind_vector)
        k3 = self.compute_derivatives(self.state + 0.5 * dt * k2, thrusts, wind_vector)
        k4 = self.compute_derivatives(self.state + dt * k3, thrusts, wind_vector)

        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Normalize yaw angle to [-pi, pi]
        self.state[8] = (self.state[8] + np.pi) % (2.0 * np.pi) - np.pi
        return self.state.copy()

    def setpoint_to_motor_thrusts(self, target_vel_world: np.ndarray, target_yaw_rate: float) -> np.ndarray:
        """
        Low-level flight controller: converts high-level velocity and yaw-rate setpoints
        into 4 motor thrust values. This represents the autonomous flight controller layer.
        """
        vel_err = target_vel_world - self.state[3:6]
        angles = self.state[6:9]
        rates = self.state[9:12]
        yaw = angles[2]

        # Desired horizontal acceleration in world frame
        acc_des = self.kp_vel * vel_err

        # Rotate horizontal acceleration into desired roll and pitch
        cy, sy = np.cos(yaw), np.sin(yaw)
        acc_body_x = cy * acc_des[0] + sy * acc_des[1]
        acc_body_y = -sy * acc_des[0] + cy * acc_des[1]

        # Tilt angles for horizontal motion (clamped for flight stability)
        max_tilt = np.deg2rad(25.0)
        des_pitch = np.clip(acc_body_x / self.gravity, -max_tilt, max_tilt)
        des_roll = np.clip(-acc_body_y / self.gravity, -max_tilt, max_tilt)

        # Vertical thrust command
        total_thrust = self.mass * (self.gravity + acc_des[2])
        # Compensate for tilt
        total_thrust /= max(0.5, np.cos(angles[0]) * np.cos(angles[1]))

        # Attitude error
        att_err = np.array([des_roll - angles[0], des_pitch - angles[1], 0.0], dtype=np.float64)
        target_rates = self.kp_ang * att_err
        target_rates[2] = target_yaw_rate

        # Angular rate error -> Torques
        rate_err = target_rates - rates
        tau_cmd = self.kd_ang * rate_err

        # Control allocation: Thrust and moments to 4 motor thrusts
        base_t = total_thrust / 4.0
        d_roll = tau_cmd[0] / (2.0 * self.arm_length)
        d_pitch = tau_cmd[1] / (2.0 * self.arm_length)
        d_yaw = tau_cmd[2] / (4.0 * self.km)

        m1 = base_t + d_pitch + d_yaw
        m2 = base_t - d_roll - d_yaw
        m3 = base_t - d_pitch + d_yaw
        m4 = base_t + d_roll - d_yaw

        motor_thrusts = np.array([m1, m2, m3, m4], dtype=np.float64)
        return np.clip(motor_thrusts, self.min_motor_thrust, self.max_motor_thrust)

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def attitude_euler(self) -> np.ndarray:
        return self.state[6:9].copy()

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.state[9:12].copy()
