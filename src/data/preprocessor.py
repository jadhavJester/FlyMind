"""
Preprocessing pipeline for BioFlight (Phase 5).
Transforms raw fruit fly flight trajectories into normalized, leak-free train/val/test splits.
Extracts tracking states, lookahead reference targets, and control target vectors.
"""

import os
import json
import yaml
from typing import Dict, List, Tuple
import numpy as np

from src.data.flight_dataset import FlightDataset


class FlightDataPreprocessor:
    """Preprocesses flight trajectories for neural flight imitation learning."""

    def __init__(self, config_path: str = "configs/dataset.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.lookahead_steps = self.cfg["features"].get("lookahead_steps", 5)
        self.seed = self.cfg["splitting"].get("random_seed", 42)

    def process(self) -> Dict[str, np.ndarray]:
        raw_h5 = self.cfg["data"]["raw_hdf5_path"]
        raw_wp = self.cfg["data"]["wing_pattern_path"]
        dataset = FlightDataset(raw_h5, raw_wp)

        # 1. Stratified Trajectory-Level Split (Guarantees zero temporal data leakage)
        train_trajs, val_trajs, test_trajs = self._split_trajectories(dataset)
        print(f"Dataset split by trajectory: Train={len(train_trajs)}, Val={len(val_trajs)}, Test={len(test_trajs)}")

        # 2. Extract state-action pairs
        X_train, Y_train = self._extract_features(train_trajs)
        X_val, Y_val = self._extract_features(val_trajs)
        X_test, Y_test = self._extract_features(test_trajs)

        # 3. Compute Normalization Statistics strictly on Train Set
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std < 1e-6] = 1.0  # Prevent division by zero

        y_mean = np.mean(Y_train, axis=0)
        y_std = np.std(Y_train, axis=0)
        y_std[y_std < 1e-6] = 1.0

        # Apply normalization
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std

        processed_data = {
            "X_train": X_train_norm.astype(np.float32),
            "Y_train": Y_train.astype(np.float32),
            "X_val": X_val_norm.astype(np.float32),
            "Y_val": Y_val.astype(np.float32),
            "X_test": X_test_norm.astype(np.float32),
            "Y_test": Y_test.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "y_mean": y_mean.astype(np.float32),
            "y_std": y_std.astype(np.float32),
        }

        # 4. Save to disk
        out_dir = self.cfg["data"]["processed_output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, self.cfg["data"]["processed_filename"])
        np.savez_compressed(npz_path, **processed_data)
        print(f"Saved processed dataset to {npz_path} ({os.path.getsize(npz_path) / 1024 / 1024:.2f} MB)")

        # Save metadata
        meta = {
            "total_samples": int(len(X_train) + len(X_val) + len(X_test)),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "input_dim": int(X_train.shape[1]),
            "target_dim": int(Y_train.shape[1]),
            "lookahead_steps": self.lookahead_steps,
            "mean": mean.tolist(),
            "std": std.tolist()
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved preprocessing metadata to {meta_path}")

        return processed_data

    def _split_trajectories(self, dataset: FlightDataset):
        """Splits trajectory objects across train, val, and test stratified by type."""
        rng = np.random.RandomState(self.seed)
        types = {}
        for traj in dataset.trajectories.values():
            types.setdefault(traj.traj_type, []).append(traj)

        train_trajs, val_trajs, test_trajs = [], [], []
        train_ratio = self.cfg["splitting"]["train_ratio"]
        val_ratio = self.cfg["splitting"]["val_ratio"]

        for t_type, trajs in types.items():
            perm = rng.permutation(trajs)
            n = len(perm)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_trajs.extend(perm[:n_train])
            val_trajs.extend(perm[n_train : n_train + n_val])
            test_trajs.extend(perm[n_train + n_val :])

        return train_trajs, val_trajs, test_trajs

    def _extract_features(self, trajectories) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts kinematic state features and control target vectors from trajectories.
        Feature vector:
        - Current linear velocity [vx, vy, vz] (3)
        - Current angular velocity [wx, wy, wz] (3)
        - Current attitude quaternion [qw, qx, qy, qz] (4)
        - Current Euler angles [roll, pitch, yaw] (3)
        - Future lookahead relative displacements [dx_1, dy_1, dz_1, ..., dx_H, dy_H, dz_H] (3 * H)
        Total input dimension = 3 + 3 + 4 + 3 + 3*5 = 28
        
        Target vector (12-dim flight control):
        - Target wing modulation, head steering, abdominal trim scaled to [-3, 3]
        """
        X_list = []
        Y_list = []
        H = self.lookahead_steps

        for traj in trajectories:
            n_frames = traj.num_frames
            if n_frames <= H + 1:
                continue

            pos = traj.position_cm  # cm
            vel = traj.linear_velocity_cm_s  # cm/s
            ang_vel = traj.angular_velocity  # rad/s
            quat = traj.quaternion
            euler = traj.euler_rad

            # Kinematic acceleration & rate targets
            acc = traj.linear_acceleration  # m/s²
            ang_acc = np.gradient(ang_vel, traj.dt, axis=0)

            for t in range(n_frames - H):
                # 1. Input Features
                cur_vel = vel[t]
                cur_ang_vel = ang_vel[t]
                cur_quat = quat[t]
                cur_euler = euler[t]

                # Lookahead relative displacement vectors: (pos[t + k] - pos[t])
                future_disp = []
                for k in range(1, H + 1):
                    future_disp.extend(pos[t + k] - pos[t])

                feature = np.concatenate([
                    cur_vel,
                    cur_ang_vel,
                    cur_quat,
                    cur_euler,
                    np.array(future_disp, dtype=np.float32)
                ])
                X_list.append(feature)

                # 2. Control Target (12-dim flight action representation)
                # Map physical maneuvers to FlyBody's 12 action channels:
                # - Wing modulation (yaw, roll, pitch for left & right wings): indices 3..8
                # - Head attitude (abduct, twist, pitch): indices 0..2
                # - Abdomen (abduct, pitch): indices 9..10
                # - WBPG freq/gain bias: index 11
                target = np.zeros(12, dtype=np.float32)
                
                # Banked turn: roll torque and yaw rate map directly to asymmetric wing differential
                roll_target = float(cur_euler[0])
                yaw_rate_target = float(cur_ang_vel[2])
                pitch_target = float(cur_euler[1])

                # Asymmetric wing modulation
                target[3] = np.clip(-0.5 * yaw_rate_target, -1.0, 1.0)   # wing_yaw_left
                target[4] = np.clip(0.8 * roll_target, -1.0, 1.0)        # wing_roll_left
                target[5] = np.clip(0.5 * pitch_target, -1.0, 1.0)       # wing_pitch_left
                target[6] = np.clip(0.5 * yaw_rate_target, -1.0, 1.0)    # wing_yaw_right
                target[7] = np.clip(-0.8 * roll_target, -1.0, 1.0)       # wing_roll_right
                target[8] = np.clip(0.5 * pitch_target, -1.0, 1.0)       # wing_pitch_right

                # Head stabilization (compensating roll/pitch)
                target[0] = np.clip(-0.2 * cur_euler[0], -0.2, 0.2)      # head_abduct
                target[1] = np.clip(-0.5 * cur_ang_vel[0], -1.0, 1.0)    # head_twist
                target[2] = np.clip(-0.3 * cur_euler[1], -0.5, 0.3)      # head pitch

                # Abdomen pitch trim
                target[9] = np.clip(0.2 * cur_euler[0], -0.7, 0.7)       # abdomen_abduct
                target[10] = np.clip(-0.4 * cur_euler[1], -1.05, 0.7)    # abdomen pitch

                # WBPG frequency / thrust modulation based on vertical heave acceleration
                target[11] = np.clip(0.1 * acc[t, 2], -1.0, 1.0)

                Y_list.append(target)

        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)
