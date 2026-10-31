"""
Winner-Take-All (WTA) Lateral Inhibition Gap Selector (Paper 2, Schoepe et al., Nature Comms 2024).

Biophysical Mechanism:
Insect central complex (fan-shaped body / ellipsoid body) features ring attractor architectures with mutual lateral inhibition.
When flying through cluttered canopies or apertures:
    1. Optical flow magnitude or nearness is accumulated across angular azimuthal sectors m in [1..M].
    2. Reciprocal lateral inhibition sharpens contrast between obstacle edges and open passages.
    3. The network performs a soft-min / argmin selection identifying the safest, obstacle-free gap heading.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class GapDecision:
    """Steering decision produced by the WTA network."""
    best_sector_idx: int
    target_azimuth_rad: float       # Desired steering angle relative to drone heading [-pi/2, pi/2]
    sector_confidence: float        # Normalized gap clarity [0.0, 1.0] (high = clear passage, low = blocked)
    inhibited_energies: np.ndarray  # Post-lateral inhibition profile across sectors
    raw_energies: np.ndarray        # Pre-inhibition flow/nearness histogram


class WinnerTakeAllGapSelector:
    """
    Bio-inspired lateral inhibition network selecting the most open flight corridor.
    """
    def __init__(
        self,
        num_sectors: int = 8,
        fov_deg: float = 90.0,
        lateral_inhibition_strength: float = 0.35,
        heading_smoothing: float = 0.60,
    ):
        self.num_sectors = num_sectors
        self.fov_rad = np.radians(fov_deg)
        self.lateral_inhibition_strength = lateral_inhibition_strength
        self.heading_smoothing = heading_smoothing

        # Sector center angles in radians relative to camera optical axis
        # e.g. for 8 sectors in 90 deg FOV, angles range symmetrically from -45 to +45 deg
        half_fov = self.fov_rad / 2.0
        sector_step = self.fov_rad / num_sectors
        self.sector_angles = np.array([
            -half_fov + (i + 0.5) * sector_step for i in range(num_sectors)
        ], dtype=np.float32)

        self._prev_target_azimuth: float = 0.0

    def reset(self) -> None:
        self._prev_target_azimuth = 0.0

    def select_gap(self, sector_obstacle_density: np.ndarray) -> GapDecision:
        """
        Identify safest opening given obstacle density (flow magnitude or nearness) across sectors.
        Args:
            sector_obstacle_density: Array of shape (num_sectors,) indicating clutter/threat in each sector.
        Returns:
            GapDecision containing desired relative yaw heading and confidence.
        """
        raw_energies = np.array(sector_obstacle_density, dtype=np.float32)

        # Invert: we seek minimum obstacle density (maximum clearance)
        # Add small epsilon to avoid division by zero
        clearance = 1.0 / (raw_energies + 0.05)

        # Lateral inhibition ring network:
        # A_i = C_i - beta * sum_{j != i} C_j
        total_clearance = np.sum(clearance)
        inhibited = clearance - self.lateral_inhibition_strength * (total_clearance - clearance) / (self.num_sectors - 1)
        inhibited = np.maximum(0.0, inhibited)

        # WTA selection
        best_idx = int(np.argmax(inhibited))
        raw_target_azimuth = float(self.sector_angles[best_idx])

        # Smooth heading transition to prevent high-frequency chattering between adjacent sectors
        smoothed_azimuth = (
            self.heading_smoothing * self._prev_target_azimuth +
            (1.0 - self.heading_smoothing) * raw_target_azimuth
        )
        self._prev_target_azimuth = smoothed_azimuth

        # Confidence: ratio of winning sector clearance over second-best / average
        sum_inh = np.sum(inhibited) + 1e-6
        confidence = float(np.clip(inhibited[best_idx] / sum_inh * 2.0, 0.0, 1.0))

        return GapDecision(
            best_sector_idx=best_idx,
            target_azimuth_rad=smoothed_azimuth,
            sector_confidence=confidence,
            inhibited_energies=inhibited,
            raw_energies=raw_energies,
        )
