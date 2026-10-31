"""
Flight dataset loader and trajectory utilities for BioFlight.
Loads high-speed Drosophila flight saccade and evasion trajectories from HDF5.
Extracts positions, linear/angular velocities, quaternions, Euler angles, and wing beat patterns.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternions [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
    Assumes aerospace intrinsic rotation sequence (z-y'-x' or standard MuJoCo convention).
    
    Args:
        q: Array of shape (..., 4) where quaternion order is [w, x, y, z].
        
    Returns:
        euler: Array of shape (..., 3) [roll, pitch, yaw] in radians.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


class FlightTrajectory:
    """Represents a single continuous fruit fly flight trajectory."""

    def __init__(
        self,
        traj_id: str,
        com_qpos: np.ndarray,
        com_qvel: np.ndarray,
        traj_type: str,
        dt: float = 0.0002
    ):
        self.traj_id = traj_id
        self.dt = dt
        self.traj_type = traj_type
        self.num_frames = len(com_qpos)
        self.duration = self.num_frames * self.dt
        self.timestamps = np.arange(self.num_frames) * self.dt

        # Positional kinematics: native FlyBody units are centimeters (cm)
        self.position_cm = com_qpos[:, 0:3]  # [x, y, z] in cm
        self.position = self.position_cm * 0.01  # [x, y, z] in meters

        # Orientation: quaternions [w, x, y, z]
        self.quaternion = com_qpos[:, 3:7]

        # Orientation: Euler angles [roll, pitch, yaw] (radians and degrees)
        self.euler_rad = quat_to_euler(self.quaternion)
        self.euler_deg = np.rad2deg(self.euler_rad)

        # Velocities: native units are cm/s and rad/s
        self.linear_velocity_cm_s = com_qvel[:, 0:3]
        self.linear_velocity = self.linear_velocity_cm_s * 0.01  # m/s
        self.angular_velocity = com_qvel[:, 3:6]  # rad/s
        self.speed = np.linalg.norm(self.linear_velocity, axis=-1)  # m/s
        self.speed_cm_s = np.linalg.norm(self.linear_velocity_cm_s, axis=-1)  # cm/s

        # Numerical accelerations (central difference)
        self.linear_acceleration = np.gradient(self.linear_velocity, self.dt, axis=0)  # m/s²
        self.accel_magnitude = np.linalg.norm(self.linear_acceleration, axis=-1)  # m/s²

    def summary(self) -> Dict[str, Union[float, int, str]]:
        return {
            "traj_id": self.traj_id,
            "traj_type": self.traj_type,
            "num_frames": self.num_frames,
            "duration_s": float(self.duration),
            "max_speed_mps": float(np.max(self.speed)),
            "mean_speed_mps": float(np.mean(self.speed)),
            "max_speed_cm_s": float(np.max(self.speed_cm_s)),
            "max_accel_mps2": float(np.max(self.accel_magnitude)),
            "max_roll_deg": float(np.max(np.abs(self.euler_deg[:, 0]))),
            "max_yaw_rate_rads": float(np.max(np.abs(self.angular_velocity[:, 2]))),
        }


class FlightDataset:
    """Manages the complete collection of flight trajectories."""

    def __init__(self, hdf5_path: str, wing_pattern_path: Optional[str] = None):
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Dataset not found at {hdf5_path}")

        self.path = hdf5_path
        self.wing_pattern_path = wing_pattern_path
        self.trajectories: Dict[str, FlightTrajectory] = {}
        self.traj_keys: List[str] = []
        self.type_indices: Dict[str, List[int]] = {}

        self._load_hdf5()
        self._load_wing_pattern()

    def _load_hdf5(self):
        with h5py.File(self.path, 'r') as f:
            self.dt = float(f['timestep_seconds'][()])
            self.freq_hz = 1.0 / self.dt
            
            # Load type indices if available
            if 'trajectory_type_indices' in f:
                for k in f['trajectory_type_indices'].keys():
                    self.type_indices[k] = list(f['trajectory_type_indices'][k][:])

            traj_group = f['trajectories']
            self.traj_keys = sorted(list(traj_group.keys()))
            for traj_key in self.traj_keys:
                item = traj_group[traj_key]
                qpos = item['com_qpos'][:]
                qvel = item['com_qvel'][:]
                raw_type = item['trajectory_type'][()]
                t_type = raw_type.decode('utf-8') if isinstance(raw_type, bytes) else str(raw_type)

                self.trajectories[traj_key] = FlightTrajectory(
                    traj_id=traj_key,
                    com_qpos=qpos,
                    com_qvel=qvel,
                    traj_type=t_type,
                    dt=self.dt
                )

    def _load_wing_pattern(self):
        if self.wing_pattern_path and os.path.exists(self.wing_pattern_path):
            self.wing_pattern = np.load(self.wing_pattern_path)
        else:
            self.wing_pattern = None

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: Union[int, str]) -> FlightTrajectory:
        if isinstance(idx, int):
            key = self.traj_keys[idx]
        else:
            key = idx
        return self.trajectories[key]

    def get_summary_dataframe(self):
        """Returns summary statistics as a list of dicts."""
        return [t.summary() for t in self.trajectories.values()]
