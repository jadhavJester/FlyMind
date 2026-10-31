"""
Conventional Computer Vision Baseline: Canny Edge Density Sector Steerer.
Represents standard classical obstacle-avoidance heuristics prior to bio-inspired or neural methods.
"""
from typing import Tuple
import cv2
import numpy as np

from ..navigation.safe_navigation.reactive_navigator import FlightSetpoint
from ..navigation.safe_navigation.reflex_state_machine import FlightMode


class CannyEdgeBaseline:
    """
    Classical vision baseline:
    Detects obstacle outlines via Canny edge filter, accumulates edge pixel counts across sectors,
    and steers toward the sector with the lowest edge density.
    """
    def __init__(
        self,
        num_sectors: int = 8,
        low_thresh: int = 50,
        high_thresh: int = 150,
        nominal_speed: float = 1.5,
    ):
        self.num_sectors = num_sectors
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.nominal_speed = nominal_speed

    def reset(self) -> None:
        pass

    def compute_setpoint(self, image: np.ndarray, current_altitude: float) -> FlightSetpoint:
        """
        Compute velocity and yaw setpoint using Canny edge distribution.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        edges = cv2.Canny(gray, self.low_thresh, self.high_thresh)
        h, w = edges.shape

        sector_w = w // self.num_sectors
        edge_counts = np.zeros(self.num_sectors, dtype=np.float32)

        for i in range(self.num_sectors):
            xs = i * sector_w
            xe = (i + 1) * sector_w if i < self.num_sectors - 1 else w
            edge_counts[i] = float(np.sum(edges[:, xs:xe] > 0))

        # Sector center angles [-45 deg, +45 deg]
        angles = np.linspace(-np.radians(45), np.radians(45), self.num_sectors)

        best_sector = int(np.argmin(edge_counts))
        worst_sector = int(np.argmax(edge_counts))
        total_edges = float(np.sum(edge_counts))

        target_yaw_rate = float(np.clip(angles[best_sector] * 1.5, -2.0, 2.0))

        # Speed throttle if whole image is saturated with edges
        max_possible_edges = (h * w) * 0.15
        edge_ratio = min(1.0, total_edges / max(1.0, max_possible_edges))
        vx = max(0.2, self.nominal_speed * (1.0 - edge_ratio))

        alt_err = 1.5 - current_altitude
        vz = float(np.clip(-1.0 * alt_err, -0.8, 0.8))

        return FlightSetpoint(
            vx_body=vx,
            vy_body=0.0,
            vz_body=vz,
            yaw_rate=target_yaw_rate,
            mode=FlightMode.CRUISE,
        )
