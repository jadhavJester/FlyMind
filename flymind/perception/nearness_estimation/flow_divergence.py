"""
Nearness Map Extraction from Optical Flow Divergence (Paper 4, Neurocomputing 2024).

Theoretical Foundation:
Nearness mu(x) is defined as reciprocal distance mu(x) = 1 / Z(x) in m^-1.
Under predominantly forward translation at velocity vz, the 2D optical flow field expands from the focus of expansion:
    u(x, y) = vz * mu(x, y) * (x - x_foe)
    v(x, y) = vz * mu(x, y) * (y - y_foe)

Taking the 2D spatial divergence:
    div(u) = d(u)/dx + d(v)/dy = 2 * vz * mu(x, y) + vz * (x * d(mu)/dx + y * d(mu)/dy)

Assuming locally piecewise-smooth obstacle surfaces:
    mu_hat(x, y) = div(u) / (2 * vz)

Benefits:
1. Purely monocular: requires zero active depth sensors (no LiDAR, no stereo baseline).
2. Inherently invariant to uniform laminar background motion.
3. Maps directly to spatial collision clearance for navigation.
"""
from typing import Optional, Tuple
import cv2
import numpy as np


class FlowDivergenceNearness:
    """
    Computes reciprocal distance (nearness) map from 2D optical flow vector fields.
    """
    def __init__(
        self,
        min_forward_speed: float = 0.2,
        smoothing_sigma: float = 2.0,
        max_nearness: float = 5.0,  # 1 / 0.2m = 5 m^-1
    ):
        self.min_forward_speed = min_forward_speed
        self.smoothing_sigma = smoothing_sigma
        self.max_nearness = max_nearness

    def compute(
        self,
        flow_u: np.ndarray,
        flow_v: np.ndarray,
        forward_speed: float,
        dt: float = 0.02,
    ) -> np.ndarray:
        """
        Compute relative nearness map from optical flow.
        Args:
            flow_u: Horizontal flow (H, W) in pixels/frame.
            flow_v: Vertical flow (H, W) in pixels/frame.
            forward_speed: Drone forward velocity vz in m/s.
            dt: Timestep between frames (s).
        Returns:
            nearness_map: Estimated reciprocal distance (H, W) in m^-1 (higher = closer obstacle).
        """
        # Convert pixels/frame to pixels/second
        flow_u_s = flow_u / max(1e-4, dt)
        flow_v_s = flow_v / max(1e-4, dt)

        # Pre-smooth flow field with Gaussian kernel to mitigate differentiation noise
        if self.smoothing_sigma > 0:
            flow_u_s = cv2.GaussianBlur(flow_u_s, (0, 0), sigmaX=self.smoothing_sigma)
            flow_v_s = cv2.GaussianBlur(flow_v_s, (0, 0), sigmaX=self.smoothing_sigma)

        # Compute central-difference spatial derivatives
        # du/dx: horizontal gradient along columns (axis=1)
        # dv/dy: vertical gradient along rows (axis=0)
        du_dx = cv2.Sobel(flow_u_s, cv2.CV_32F, 1, 0, ksize=3)
        dv_dy = cv2.Sobel(flow_v_s, cv2.CV_32F, 0, 1, ksize=3)

        divergence = du_dx + dv_dy

        # Normalize by forward velocity 2 * vz
        effective_vz = max(self.min_forward_speed, abs(forward_speed))
        raw_nearness = divergence / (2.0 * effective_vz)

        # Negative divergence corresponds to receding surfaces; clip to positive looming obstacles
        nearness = np.clip(raw_nearness, 0.0, self.max_nearness)

        return nearness.astype(np.float32)

    def compute_sector_nearness(self, nearness_map: np.ndarray, num_sectors: int = 8) -> np.ndarray:
        """
        Partition nearness map into horizontal azimuthal sectors for navigation gap finding.
        Returns array of shape (num_sectors,) with mean nearness per sector.
        """
        h, w = nearness_map.shape
        sector_width = w // num_sectors
        sectors = np.zeros(num_sectors, dtype=np.float32)

        for i in range(num_sectors):
            x_start = i * sector_width
            x_end = (i + 1) * sector_width if i < num_sectors - 1 else w
            sectors[i] = float(np.mean(nearness_map[:, x_start:x_end]))

        return sectors
