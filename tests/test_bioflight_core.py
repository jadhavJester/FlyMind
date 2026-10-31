"""
Core unit tests for BioFlight (Section 27).
Tests:
- Dataset loading
- Preprocessing and normalization
- Model input/output dimensions
- Environment reset and step
- Action validity and bounds
- Quantitative evaluation metrics
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.flight_dataset import FlightDataset, quat_to_euler
from src.models.mlp_policy import MLPPolicy
from src.evaluation.metrics import compute_flight_metrics
from flybody.fly_envs import flight_imitation


def test_quat_to_euler():
    """Verify quaternion to Euler angle conversion."""
    # Identity quaternion [w=1, x=0, y=0, z=0] -> [roll=0, pitch=0, yaw=0]
    q_ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    euler = quat_to_euler(q_ident)
    assert np.allclose(euler, 0.0, atol=1e-5), f"Expected [0, 0, 0], got {euler}"


def test_dataset_loading():
    """Verify loading trajectories from raw HDF5 dataset."""
    raw_h5 = os.path.join("data", "raw", "flight-dataset_saccade-evasion_augmented.hdf5")
    raw_wp = os.path.join("data", "raw", "wing_pattern_fmech.npy")
    dataset = FlightDataset(raw_h5, raw_wp)

    assert len(dataset) == 272, f"Expected 272 trajectories, got {len(dataset)}"
    assert np.isclose(dataset.dt, 0.0002), f"Expected dt=0.0002, got {dataset.dt}"
    assert dataset.freq_hz == 5000.0

    traj0 = dataset[0]
    assert traj0.position.shape[1] == 3
    assert traj0.quaternion.shape[1] == 4
    assert traj0.linear_velocity.shape[1] == 3
    assert traj0.angular_velocity.shape[1] == 3
    assert not np.isnan(traj0.position).any()


def test_preprocessing_and_normalization():
    """Verify preprocessed data files and zero-leakage normalization."""
    proc_file = os.path.join("data", "processed", "flight_data_processed.npz")
    assert os.path.exists(proc_file), "Processed dataset npz missing!"

    data = np.load(proc_file)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]

    assert X_train.shape[1] == 28, f"Expected input dim 28, got {X_train.shape[1]}"
    assert Y_train.shape[1] == 12, f"Expected target dim 12, got {Y_train.shape[1]}"
    assert not np.isnan(X_train).any()
    assert not np.isnan(Y_train).any()

    # Verify normalization on training set: mean ~ 0, std ~ 1
    assert np.allclose(np.mean(X_train, axis=0), 0.0, atol=1e-1)
    assert np.allclose(np.std(X_train, axis=0), 1.0, atol=1e-1)


def test_model_input_output_dimensions():
    """Verify MLPPolicy architecture forward pass and action bounds."""
    model = MLPPolicy(input_dim=28, action_dim=12, hidden_dims=[256, 256, 128], action_scale=3.0)
    batch_size = 16
    dummy_input = torch.randn(batch_size, 28)
    output = model(dummy_input)

    assert output.shape == (batch_size, 12)
    assert torch.all(output >= -3.0) and torch.all(output <= 3.0), "Actions exceeded bounds [-3, 3]"

    # Test single-step inference
    single_obs = np.random.randn(28).astype(np.float32)
    act = model.select_action(single_obs)
    assert act.shape == (12,)
    assert np.all(act >= -3.0) and np.all(act <= 3.0)


def test_environment_reset_and_step():
    """Verify FlyBody environment reset, stepping, and action space."""
    env = flight_imitation(
        ref_path="data/raw/flight-dataset_saccade-evasion_augmented.hdf5",
        wpg_pattern_path="data/raw/wing_pattern_fmech.npy",
        randomize_start_step=False
    )
    timestep = env.reset()
    assert timestep.step_type is not None
    assert timestep.observation is not None

    action_spec = env.action_spec()
    assert action_spec.shape == (12,)

    # Test valid action step
    action = np.zeros(12, dtype=np.float32)
    next_ts = env.step(action)
    assert next_ts.reward is not None
    assert next_ts.observation is not None


def test_evaluation_metrics():
    """Verify computation of flight trajectory metrics."""
    T = 50
    pos_act = np.zeros((T, 3))
    pos_tgt = np.ones((T, 3)) * 0.1  # constant 0.1 cm offset
    quat_act = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    quat_tgt = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    vel_act = np.zeros((T, 3))
    vel_tgt = np.zeros((T, 3))
    actions = np.zeros((T, 12))

    metrics = compute_flight_metrics(
        pos_actual=pos_act,
        pos_target=pos_tgt,
        quat_actual=quat_act,
        quat_target=quat_tgt,
        vel_actual=vel_act,
        vel_target=vel_tgt,
        actions=actions,
        dt=0.0002,
        terminated_early=False,
        max_time=0.01
    )

    assert "pos_rmse" in metrics
    assert "roll_error_deg" in metrics
    assert "control_effort" in metrics
    assert np.isclose(metrics["pos_rmse"], np.linalg.norm([0.1, 0.1, 0.1]))
    assert metrics["roll_error_deg"] == 0.0
    assert metrics["control_effort"] == 0.0
