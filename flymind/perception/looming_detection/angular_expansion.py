"""
Angular Expansion and Looming Detection Module.
Computes optical expansion rate dtheta/dt, subtended angle theta, and Time-To-Contact (TTC).

Formulation:
- A looming hazard of radius R approaching at speed v has angular size theta(t) ~ 2 * arctan(R / D(t)).
- In small-angle approximation: theta(t) ~ 2R / D(t).
- Expansion rate: dtheta/dt = 2R * v / D(t)^2 = (v / D(t)) * theta(t).
- Time-To-Contact: tau(t) = theta(t) / (dtheta/dt) = D(t) / v.
- Characteristic looming activation: eta(t) = theta(t) * exp( -alpha * (tau(t) - tau_c) ).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class LoomingThreat:
    """Represents an identified looming obstacle."""
    center_x: float          # Normalized image coordinate [-1.0, 1.0] (azimuth)
    center_y: float          # Normalized image coordinate [-1.0, 1.0] (elevation)
    theta_rad: float         # Subtended angular size in radians
    theta_dot_rad_s: float   # Expansion rate in rad/s
    ttc_seconds: float       # Time-to-contact in seconds (inf if not looming)
    collision_risk: float    # Normalized risk score [0.0, 1.0]


@dataclass
class LoomingState:
    """Overall looming assessment across visual field."""
    max_risk: float
    min_ttc: float
    dominant_threat: Optional[LoomingThreat]
    all_threats: List[LoomingThreat]
    is_imminent_danger: bool


class LoomingDetector:
    """
    Bio-inspired looming obstacle detector.
    Analyzes visual blob expansion and diverging optical flow fields.
    """
    def __init__(
        self,
        fov_deg: float = 90.0,
        min_blob_area: int = 50,
        ttc_threshold_s: float = 1.5,
        risk_threshold: float = 0.65,
    ):
        self.fov_rad = np.radians(fov_deg)
        self.min_blob_area = min_blob_area
        self.ttc_threshold_s = ttc_threshold_s
        self.risk_threshold = risk_threshold

        self._prev_blobs: List[Tuple[float, float, float]] = []  # (cx, cy, radius)
        self._prev_time: Optional[float] = None

    def reset(self) -> None:
        self._prev_blobs.clear()
        self._prev_time = None

    def detect_looming(
        self,
        image: np.ndarray,
        flow_u: np.ndarray,
        flow_v: np.ndarray,
        timestamp: float,
    ) -> LoomingState:
        """
        Detect looming threats from current visual frame and optical flow field.
        Args:
            image: Visual frame (H, W) or (H, W, 3).
            flow_u: Horizontal flow (pixels/frame).
            flow_v: Vertical flow (pixels/frame).
            timestamp: Timestamp in seconds.
        """
        h, w = image.shape[:2]
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Invert or threshold dark obstacles against sky/ground background
        # Drosophila reacts strongly to dark approaching objects (OFF pathway)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_blobs: List[Tuple[float, float, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_blob_area:
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                current_blobs.append((cx, cy, radius))

        threats: List[LoomingThreat] = []
        dt = 0.05
        if self._prev_time is not None and timestamp > self._prev_time:
            dt = max(0.001, timestamp - self._prev_time)

        # Match blobs with previous frame to measure radius expansion dr/dt
        for cx, cy, r in current_blobs:
            # Estimate expansion from optical flow divergence within blob
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)

            # Subtended angle in radians: theta = 2 * r / (focal_length_px)
            # focal length f_px = (w / 2) / tan(fov / 2)
            f_px = (w / 2.0) / np.tan(self.fov_rad / 2.0)
            theta = 2.0 * np.arctan(r / f_px)

            # Match with closest previous blob
            matched_r_prev = None
            min_dist = 50.0  # max tracking pixel displacement
            for px, py, pr in self._prev_blobs:
                dist = np.hypot(cx - px, cy - py)
                if dist < min_dist:
                    min_dist = dist
                    matched_r_prev = pr

            if matched_r_prev is not None and matched_r_prev > 0:
                dr_dt = (r - matched_r_prev) / dt
                theta_prev = 2.0 * np.arctan(matched_r_prev / f_px)
                theta_dot = max(0.0, (theta - theta_prev) / dt)
            else:
                # Approximate theta_dot from flow divergence around perimeter
                u_in_blob = flow_u[mask > 0]
                v_in_blob = flow_v[mask > 0]
                flow_mag = float(np.mean(np.hypot(u_in_blob, v_in_blob))) if len(u_in_blob) > 0 else 0.0
                theta_dot = (flow_mag / f_px) / dt if f_px > 0 else 0.0

            # Compute TTC
            if theta_dot > 1e-4:
                ttc = float(np.clip(theta / theta_dot, 0.05, 10.0))
            else:
                ttc = 99.9

            # Collision risk metric: rises sharply as theta grows AND ttc drops
            # eta = (theta / 0.5) * exp(-ttc / 1.0)
            norm_theta = min(1.0, theta / (np.pi / 3.0))
            risk = float(np.clip(norm_theta * np.exp(-ttc / 1.5) * 2.5, 0.0, 1.0))

            # Convert to normalized visual coordinates [-1, 1]
            norm_x = (cx - w / 2.0) / (w / 2.0)
            norm_y = (cy - h / 2.0) / (h / 2.0)

            threats.append(
                LoomingThreat(
                    center_x=norm_x,
                    center_y=norm_y,
                    theta_rad=theta,
                    theta_dot_rad_s=theta_dot,
                    ttc_seconds=ttc,
                    collision_risk=risk,
                )
            )

        self._prev_blobs = current_blobs
        self._prev_time = timestamp

        # Summarize state
        if threats:
            threats.sort(key=lambda t: t.collision_risk, reverse=True)
            dominant = threats[0]
            max_risk = dominant.collision_risk
            min_ttc = min(t.ttc_seconds for t in threats)
        else:
            dominant = None
            max_risk = 0.0
            min_ttc = 99.9

        is_danger = max_risk >= self.risk_threshold or min_ttc <= self.ttc_threshold_s

        return LoomingState(
            max_risk=max_risk,
            min_ttc=min_ttc,
            dominant_threat=dominant,
            all_threats=threats,
            is_imminent_danger=is_danger,
        )
