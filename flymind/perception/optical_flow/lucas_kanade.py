"""
Lucas-Kanade Optical Flow Estimator.

Computes sparse and semi-dense optical flow vectors using pyramidal Lucas-Kanade.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional


class LucasKanadeFlow:
    """
    Computes optical flow vectors using cv2.calcOpticalFlowPyrLK on a uniform grid of visual points.
    """

    def __init__(self, width: int = 160, height: int = 120, grid_step: int = 8):
        self.width = width
        self.height = height
        self.grid_step = grid_step
        self.prev_gray: Optional[np.ndarray] = None

        # Precompute uniform grid sample points across the frame
        xs = np.arange(grid_step // 2, width, grid_step, dtype=np.float32)
        ys = np.arange(grid_step // 2, height, grid_step, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        self.grid_points = np.column_stack([xv.ravel(), yv.ravel()]).reshape(-1, 1, 2)

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def reset(self):
        self.prev_gray = None

    def compute(self, current_frame_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes flow vectors between previous and current frame.
        Returns:
          points_prev: (N, 2) array of start coordinates
          points_curr: (N, 2) array of tracked coordinates
          flow_vectors: (N, 2) array of displacements (dx, dy)
        """
        if current_frame_rgb.ndim == 3:
            curr_gray = cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = current_frame_rgb.copy()

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            n = len(self.grid_points)
            return np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))

        # Track grid points
        p_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.grid_points, None, **self.lk_params
        )

        valid = status.ravel() == 1
        pts_prev = self.grid_points[valid].reshape(-1, 2)
        pts_curr = p_curr[valid].reshape(-1, 2)
        flow_vecs = pts_curr - pts_prev

        self.prev_gray = curr_gray
        return pts_prev, pts_curr, flow_vecs
