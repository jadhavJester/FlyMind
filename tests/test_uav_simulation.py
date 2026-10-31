"""
Unit tests for Phase 10 Quadrotor UAV Simulation & Controllers.
"""

import pytest
import numpy as np
from simulation.uav.quadrotor_dynamics import QuadrotorDynamics
from simulation.uav.quadrotor_env import QuadrotorEnv
from src.controllers.uav_controllers import ConventionalPIDQuadrotorController, BioInspiredTransferController


def test_quadrotor_dynamics_hover_equilibrium():
    dynamics = QuadrotorDynamics(mass=0.5)
    state = np.zeros(12, dtype=np.float32)
    state[2] = 1.0  # Height 1.0 m

    # Apply exact hover thrust (m * g / 4)
    hover_thrust = (0.5 * 9.81) / 4.0
    rotor_thrusts = np.array([hover_thrust] * 4, dtype=np.float32)

    d_state = dynamics.derivatives(state, rotor_thrusts)
    # Vertical acceleration should be zero
    assert abs(d_state[5]) < 1e-4


def test_quadrotor_rk4_integration():
    dynamics = QuadrotorDynamics()
    state = np.zeros(12, dtype=np.float32)
    state[2] = 1.0
    hover_thrust = (0.5 * 9.81) / 4.0

    next_state = dynamics.step_rk4(state, np.array([hover_thrust]*4), dt=0.01)
    assert not np.isnan(next_state).any()
    assert abs(next_state[2] - 1.0) < 1e-3


def test_quadrotor_controllers_output():
    pid_ctrl = ConventionalPIDQuadrotorController()
    bio_ctrl = BioInspiredTransferController()

    state = np.zeros(12, dtype=np.float32)
    state[2] = 1.0
    target = np.array([0.1, 0.1, 1.0], dtype=np.float32)

    pid_thrusts = pid_ctrl.compute_action(state, target)
    bio_thrusts = bio_ctrl.compute_action(state, target)

    assert pid_thrusts.shape == (4,)
    assert bio_thrusts.shape == (4,)
    assert (pid_thrusts >= 0.0).all()
    assert (bio_thrusts >= 0.0).all()
