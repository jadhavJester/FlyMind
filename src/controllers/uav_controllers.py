"""
UAV Quadrotor Flight Controllers for BioFlight (Phase 10).
Implements Conventional Cascaded PID Controller vs Bio-Inspired Transfer Controller.
"""

import numpy as np


class ConventionalPIDQuadrotorController:
    """
    Standard Cascaded Proportional-Integral-Derivative (PID) Controller for Quadrotor UAVs.
    Position Loop (PD) -> Desired Roll/Pitch -> Attitude Loop (PD) -> Motor Thrust Mixing.
    """

    def __init__(self, hover_thrust: float = 1.226):
        self.hover_thrust = hover_thrust  # (m * g) / 4 = (0.5 * 9.81) / 4 = 1.226 N

        # Position gains
        self.kp_pos = 1.5
        self.kd_pos = 1.2

        # Altitude gains
        self.kp_z = 8.0
        self.kd_z = 4.0

        # Attitude gains
        self.kp_att = 4.0
        self.kd_att = 1.0

    def compute_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]   # [roll, pitch, yaw]
        ang_vel = state[9:12]

        pos_err = target_pos - pos

        # Altitude control (Total thrust)
        u_z = self.kp_z * pos_err[2] - self.kd_z * vel[2]
        total_thrust = 4.0 * self.hover_thrust + u_z
        total_thrust = np.clip(total_thrust, 0.0, 10.0)

        # Desired roll and pitch from horizontal position error
        des_roll = np.clip(- (self.kp_pos * pos_err[1] - self.kd_pos * vel[1]), -0.5, 0.5)
        des_pitch = np.clip(self.kp_pos * pos_err[0] - self.kd_pos * vel[0], -0.5, 0.5)

        # Attitude loop
        roll_err = des_roll - euler[0]
        pitch_err = des_pitch - euler[1]
        yaw_err = 0.0 - euler[2]

        u_roll = self.kp_att * roll_err - self.kd_att * ang_vel[0]
        u_pitch = self.kp_att * pitch_err - self.kd_att * ang_vel[1]
        u_yaw = self.kp_att * yaw_err - self.kd_att * ang_vel[2]

        # Motor Thrust Mixing (X configuration)
        f1 = (total_thrust / 4.0) + u_roll - u_pitch - u_yaw
        f2 = (total_thrust / 4.0) + u_roll + u_pitch + u_yaw
        f3 = (total_thrust / 4.0) - u_roll + u_pitch - u_yaw
        f4 = (total_thrust / 4.0) - u_roll - u_pitch + u_yaw

        rotor_thrusts = np.clip(np.array([f1, f2, f3, f4], dtype=np.float32), 0.0, 5.0)
        return rotor_thrusts


class BioInspiredTransferController:
    """
    Bio-Inspired Transfer Controller.
    Adapts high-frequency biological fruit-fly bank-turn saccade dynamics (learned from Drosophila locomotion data)
    to quadrotor motor differential thrust control.
    """

    def __init__(self, hover_thrust: float = 1.226):
        self.hover_thrust = hover_thrust
        self.kp_pos = 2.5
        self.kd_pos = 1.5
        self.kp_bank = 6.0  # High biological roll-bank feedback gain
        self.kd_bank = 1.8

    def compute_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        ang_vel = state[9:12]

        pos_err = target_pos - pos

        # Total altitude thrust with feedforward lift compensation
        u_z = 10.0 * pos_err[2] - 4.5 * vel[2]
        total_thrust = 4.0 * self.hover_thrust + u_z

        # Biological Banked Turn Saccade Strategy:
        # Rapid lateral turns require aggressive roll banking coupled into differential thrust
        des_bank_roll = np.clip(- (self.kp_pos * pos_err[1] - self.kd_pos * vel[1]) * 1.5, -0.8, 0.8)
        des_pitch = np.clip(self.kp_pos * pos_err[0] - self.kd_pos * vel[0], -0.5, 0.5)

        roll_err = des_bank_roll - euler[0]
        pitch_err = des_pitch - euler[1]

        u_roll = self.kp_bank * roll_err - self.kd_bank * ang_vel[0]
        u_pitch = self.kp_bank * pitch_err - self.kd_bank * ang_vel[1]
        u_yaw = - 0.5 * ang_vel[2]

        # Bio-inspired differential thrust mapping
        f1 = (total_thrust / 4.0) + u_roll - u_pitch - u_yaw
        f2 = (total_thrust / 4.0) + u_roll + u_pitch + u_yaw
        f3 = (total_thrust / 4.0) - u_roll + u_pitch - u_yaw
        f4 = (total_thrust / 4.0) - u_roll - u_pitch + u_yaw

        rotor_thrusts = np.clip(np.array([f1, f2, f3, f4], dtype=np.float32), 0.0, 5.0)
        return rotor_thrusts
