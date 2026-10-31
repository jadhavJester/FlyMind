"""
ROS 2 & PX4 MAVLink Bridge Definition.

Provides standard serialization schemas matching ROS 2 Humble conventions:
- sensor_msgs/Image
- sensor_msgs/Imu
- geometry_msgs/Twist
- mavros/setpoint_velocity/cmd_vel_unstamped

Allows seamless sim-to-real transfer to Gazebo Sim and PX4 SITL without code modification.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


class ROS2BridgeMessage:
    """Standardized ROS 2 representation for FlyMind sensor and command topics."""

    @staticmethod
    def encode_image_msg(rgb_frame: np.ndarray, frame_id: str = "camera_optical_frame") -> Dict[str, Any]:
        """Encodes RGB frame into ROS 2 sensor_msgs/Image dictionary."""
        h, w, c = rgb_frame.shape
        return {
            "header": {"frame_id": frame_id, "stamp_s": 0.0},
            "height": int(h),
            "width": int(w),
            "encoding": "rgb8",
            "is_bigendian": 0,
            "step": int(w * c),
            "data": rgb_frame.tobytes()
        }

    @staticmethod
    def encode_imu_msg(gyro: np.ndarray, accel: np.ndarray, frame_id: str = "base_link") -> Dict[str, Any]:
        """Encodes gyro and accel into ROS 2 sensor_msgs/Imu dictionary."""
        return {
            "header": {"frame_id": frame_id, "stamp_s": 0.0},
            "angular_velocity": {"x": float(gyro[0]), "y": float(gyro[1]), "z": float(gyro[2])},
            "linear_acceleration": {"x": float(accel[0]), "y": float(accel[1]), "z": float(accel[2])}
        }

    @staticmethod
    def encode_twist_cmd(vel_cmd: np.ndarray, yaw_rate_cmd: float) -> Dict[str, Any]:
        """Encodes high-level navigation output into geometry_msgs/Twist dictionary."""
        return {
            "linear": {"x": float(vel_cmd[0]), "y": float(vel_cmd[1]), "z": float(vel_cmd[2])},
            "angular": {"x": 0.0, "y": 0.0, "z": float(yaw_rate_cmd)}
        }
