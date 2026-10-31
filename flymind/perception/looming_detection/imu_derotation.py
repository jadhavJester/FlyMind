"""
IMU Optical Flow Derotation Module.
Implements Longuet-Higgins & Prazdny rotational flow subtraction using 3-axis gyroscope telemetry (Zhao et al., 2024).

Theoretical formulation:
The total apparent optical flow observed on the image plane decomposes into translational and rotational components:
    u_total(x, y) = u_trans(x, y) + u_rot(x, y)

The rotational flow is completely independent of scene depth and given analytically by camera angular velocity omega = [wx, wy, wz]^T:
    u_rot(x, y) = (x*y / f)*wx - (f + x^2 / f)*wy + y*wz
    v_rot(x, y) = (f + y^2 / f)*wx - (x*y / f)*wy - x*wz

Subtracting u_rot isolates pure translational flow u_trans, preventing false looming detections caused by drone yaw/pitch/roll manoeuvres.
"""
from typing import Tuple
import numpy as np


class IMUFlowDerotator:
    """
    Subtracts ego-rotational optical flow using gyroscope rate measurements.
    Precomputes geometric projection matrices B(x, y) for real-time operation.
    """
    def __init__(self, width: int = 160, height: int = 120, fov_deg: float = 90.0):
        self.width = width
        self.height = height
        self.fov_rad = np.radians(fov_deg)

        # Camera intrinsic parameters
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0
        self.fx = (width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.fy = self.fx  # square pixels

        # Precompute coordinate grids centered at principal point
        xs = np.arange(width, dtype=np.float32) - self.cx
        ys = np.arange(height, dtype=np.float32) - self.cy
        x_grid, y_grid = np.meshgrid(xs, ys)

        # Precompute rotational sensitivity tensors B(x, y) such that [u_rot; v_rot] = B * [wx, wy, wz]^T
        # Shape: (height, width, 3)
        self.B_u = np.zeros((height, width, 3), dtype=np.float32)
        self.B_v = np.zeros((height, width, 3), dtype=np.float32)

        # u_rot = (x*y / fy)*wx - (fx + x^2 / fx)*wy + y*wz
        self.B_u[..., 0] = (x_grid * y_grid) / self.fy
        self.B_u[..., 1] = -(self.fx + (x_grid ** 2) / self.fx)
        self.B_u[..., 2] = y_grid

        # v_rot = (fy + y^2 / fy)*wx - (x*y / fx)*wy - x*wz
        self.B_v[..., 0] = self.fy + (y_grid ** 2) / self.fy
        self.B_v[..., 1] = -(x_grid * y_grid) / self.fx
        self.B_v[..., 2] = -x_grid

    def compute_rotational_flow(self, omega: np.ndarray, dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pixel displacement caused solely by camera angular rotation over interval dt.
        Args:
            omega: 3-element gyro rate vector [wx, wy, wz] in rad/s in camera frame.
                   wx = pitch rate (tilt up/down), wy = yaw rate (turn right/left), wz = roll rate.
            dt: Time interval corresponding to the flow frame step (s).
        Returns:
            u_rot: Horizontal rotational flow (H, W) in pixels.
            v_rot: Vertical rotational flow (H, W) in pixels.
        """
        # Convert rad/s to rad/frame: delta_angle = omega * dt
        d_omega = omega.astype(np.float32) * dt

        u_rot = (
            self.B_u[..., 0] * d_omega[0] +
            self.B_u[..., 1] * d_omega[1] +
            self.B_u[..., 2] * d_omega[2]
        )
        v_rot = (
            self.B_v[..., 0] * d_omega[0] +
            self.B_v[..., 1] * d_omega[1] +
            self.B_v[..., 2] * d_omega[2]
        )
        return u_rot, v_rot

    def derotate(
        self,
        flow_u: np.ndarray,
        flow_v: np.ndarray,
        omega: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtract ego-rotational component from total optical flow to recover pure translational flow.
        Args:
            flow_u: Measured total horizontal flow (H, W).
            flow_v: Measured total vertical flow (H, W).
            omega: 3D camera angular rates [wx, wy, wz] (rad/s).
            dt: Timestep (s).
        Returns:
            u_trans: Derotated horizontal flow (H, W).
            v_trans: Derotated vertical flow (H, W).
        """
        u_rot, v_rot = self.compute_rotational_flow(omega, dt=dt)
        u_trans = flow_u - u_rot
        v_trans = flow_v - v_rot
        return u_trans, v_trans
