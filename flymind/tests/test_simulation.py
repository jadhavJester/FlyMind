"""
Test Simulation Core: 6-DoF quadrotor, camera, IMU, and environments.
"""
import numpy as np
import pytest

from flymind.simulation.quadrotor_6dof import Quadrotor6DoF
from flymind.simulation.sensors.synthetic_camera import SyntheticCamera
from flymind.simulation.sensors.noisy_imu import NoisyIMU
from flymind.simulation.environments.obstacle_forest import ObstacleForest
from flymind.simulation.environments.narrow_gates import NarrowGatesCorridor
from flymind.simulation.environments.dynamic_looming import DynamicLoomingArena


def test_quadrotor_dynamics():
    quad = Quadrotor6DoF(dt=0.02, initial_pos=np.array([0.0, 0.0, 1.5]))
    assert np.allclose(quad.pos, [0.0, 0.0, 1.5])
    assert quad.yaw == 0.0

    # Command forward velocity
    quad.step_velocity_body(vx_body=1.0, vy_body=0.0, vz_body=0.0, yaw_rate_cmd=0.0)
    assert quad.pos[0] > 0.0  # moved forward in x


def test_synthetic_camera_rendering():
    cam = SyntheticCamera(width=80, height=60, fov_deg=90.0)
    env = ObstacleForest(num_obstacles=10, seed=42)
    obstacles = env.get_obstacles(0.0)

    img, depth = cam.render(np.array([0.0, 0.0, 1.5]), 0.0, obstacles)
    assert img.shape == (60, 80, 3)
    assert depth.shape == (60, 80)
    assert np.all(depth >= 0.0)


def test_noisy_imu():
    imu = NoisyIMU(dt=0.02)
    true_omega = np.array([0.1, -0.2, 0.05])
    true_accel = np.array([0.0, 0.0, -9.81])

    omega_meas, accel_meas = imu.measure(true_omega, true_accel)
    assert omega_meas.shape == (3,)
    assert accel_meas.shape == (3,)
    # Noise should be bounded reasonably
    assert np.linalg.norm(omega_meas - true_omega) < 0.5


def test_environments_clearance():
    forest = ObstacleForest(num_obstacles=15, seed=123)
    clearance = forest.check_clearance(np.array([0.0, 0.0, 1.5]))
    assert clearance > 0.0

    gates = NarrowGatesCorridor(gate_spacing=5.0)
    gate_clear = gates.check_clearance(np.array([0.0, 0.0, 1.5]))
    assert gate_clear > 0.0

    looming = DynamicLoomingArena(approach_speed=2.0)
    obs = looming.get_obstacles(0.0)
    assert len(obs) >= 1

    from flymind.simulation.environments.moving_target_track import MovingTargetTrack
    track = MovingTargetTrack(track_length_m=12.0, target_speed=1.5)
    track_obs = track.get_obstacles(0.0)
    assert len(track_obs) > 5  # Has pylons and moving target
    init_clearance = track.check_clearance(np.array([0.0, 0.0, 1.5]))
    assert init_clearance > 0.5
    # Step environment time
    track_obs_t1 = track.get_obstacles(1.0)
    target_pos = track.target.current_pos
    assert target_pos[0] > 0.0  # target exists along track
