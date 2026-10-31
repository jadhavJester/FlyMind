"""
Dense Optical Flow via Gunnar Farneback's Algorithm.
Provides continuous pixel-level velocity fields (u, v) across the drone's visual field.
"""
from typing import Optional, Tuple
import cv2
import numpy as np


class FarnebackDenseFlow:
    """
    Computes dense 2D optical flow field using OpenCV's Farneback formulation.
    Returns horizontal (u) and vertical (v) optical flow components in pixels/frame.
    """
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
        temporal_alpha: float = 0.7,
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.temporal_alpha = temporal_alpha

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_flow: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal history frames."""
        self._prev_gray = None
        self._prev_flow = None

    def compute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow between previous and current frames.
        Args:
            image: Grayscale or BGR/RGB image (H, W) or (H, W, 3).
        Returns:
            flow_u: Horizontal flow field (H, W), positive rightwards.
            flow_v: Vertical flow field (H, W), positive downwards.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            h, w = gray.shape
            self._prev_flow = np.zeros((h, w, 2), dtype=np.float32)
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            prev=self._prev_gray,
            next=gray,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        )

        # Apply exponential temporal smoothing to reduce frame-to-frame jitter
        if self._prev_flow is not None and self.temporal_alpha < 1.0:
            flow = self.temporal_alpha * flow + (1.0 - self.temporal_alpha) * self._prev_flow

        self._prev_gray = gray
        self._prev_flow = flow

        return flow[..., 0], flow[..., 1]

    @staticmethod
    def flow_to_polar(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Cartesian flow (u, v) to magnitude and orientation angle in radians [-pi, pi]."""
        magnitude = np.hypot(u, v)
        angle = np.arctan2(v, u)
        return magnitude, angle
