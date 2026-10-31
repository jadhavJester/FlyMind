"""
Quantitative evaluation metrics for BioFlight (Phase 7).
Measures tracking fidelity, orientation accuracy, flight stability,
and control effort for closed-loop flight controllers.
"""

from typing import Dict, List, Union
import numpy as np
from src.data.flight_dataset import quat_to_euler


def compute_flight_metrics(
    pos_actual: np.ndarray,      # (T, 3) in cm or m
    pos_target: np.ndarray,      # (T, 3) in cm or m
    quat_actual: np.ndarray,     # (T, 4) [w, x, y, z]
    quat_target: np.ndarray,     # (T, 4) [w, x, y, z]
    vel_actual: np.ndarray,      # (T, 3) in cm/s or m/s
    vel_target: np.ndarray,      # (T, 3) in cm/s or m/s
    actions: np.ndarray,         # (T, 12)
    dt: float,
    terminated_early: bool,
    max_time: float
) -> Dict[str, float]:
    """
    Computes standard robotic and biological flight control benchmarks.
    """
    T = len(pos_actual)
    duration = T * dt

    # 1. Trajectory Position Errors
    pos_err = pos_actual - pos_target
    pos_dist = np.linalg.norm(pos_err, axis=-1)
    pos_rmse = float(np.sqrt(np.mean(pos_dist ** 2)))
    pos_mae = float(np.mean(pos_dist))
    max_deviation = float(np.max(pos_dist))

    # 2. Velocity Tracking Errors
    vel_err = vel_actual - vel_target
    vel_rmse = float(np.sqrt(np.mean(np.sum(vel_err ** 2, axis=-1))))
    target_vel_error = float(np.mean(np.abs(vel_err)))

    # 3. Orientation & Attitude Errors
    euler_act = quat_to_euler(quat_actual)  # (T, 3) [roll, pitch, yaw] rad
    euler_tgt = quat_to_euler(quat_target)  # (T, 3)
    euler_diff = np.rad2deg(euler_act - euler_tgt)
    # Wrap angles to [-180, 180]
    euler_diff = (euler_diff + 180.0) % 360.0 - 180.0

    roll_error = float(np.mean(np.abs(euler_diff[:, 0])))
    pitch_error = float(np.mean(np.abs(euler_diff[:, 1])))
    yaw_error = float(np.mean(np.abs(euler_diff[:, 2])))
    roll_variance = float(np.var(euler_diff[:, 0]))

    # 4. Control Action Metrics
    action_mag = float(np.mean(np.abs(actions)))
    control_effort = float(np.mean(np.sum(actions ** 2, axis=-1)))
    if T > 1:
        action_diff = np.diff(actions, axis=0)
        action_smoothness = float(np.mean(np.sum(action_diff ** 2, axis=-1)))
    else:
        action_smoothness = 0.0

    # 5. Stability & Survival
    survival_fraction = float(duration / max_time)
    crash = 1.0 if terminated_early and (duration < max_time * 0.95) else 0.0

    return {
        "flight_duration_s": duration,
        "survival_fraction": survival_fraction,
        "crashed": crash,
        "pos_rmse": pos_rmse,
        "pos_mae": pos_mae,
        "max_pos_deviation": max_deviation,
        "vel_rmse": vel_rmse,
        "target_vel_error": target_vel_error,
        "roll_error_deg": roll_error,
        "pitch_error_deg": pitch_error,
        "yaw_error_deg": yaw_error,
        "roll_variance": roll_variance,
        "action_magnitude": action_mag,
        "action_smoothness": action_smoothness,
        "control_effort": control_effort
    }
