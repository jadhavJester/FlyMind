"""
Collision Risk Assessment and Spatial Risk Mapping.
Fuses looming expansion, TTC, and optical flow divergence into unified navigational threat metrics.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ...perception.looming_detection.angular_expansion import LoomingState
from ...bioinspired.lplc2.population_coding import LPLC2PopulationResponse


@dataclass
class SpatialRiskMap:
    """Multi-sector collision risk distribution."""
    sector_risks: np.ndarray       # Risk in each horizontal sector [0.0 (safe) to 1.0 (imminent collision)]
    max_risk: float
    safest_sector_idx: int
    most_dangerous_sector_idx: int
    overall_threat_level: str      # 'CLEAR', 'ADVISORY', 'WARNING', 'CRITICAL'


class CollisionRiskAssessor:
    """
    Evaluates immediate collision probability by integrating optical flow divergence,
    looming blob expansion, and LPLC2 neural activation.
    """
    def __init__(
        self,
        num_sectors: int = 8,
        advisory_threshold: float = 0.35,
        warning_threshold: float = 0.60,
        critical_threshold: float = 0.85,
    ):
        self.num_sectors = num_sectors
        self.advisory_threshold = advisory_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def evaluate(
        self,
        sector_nearness: np.ndarray,
        looming_state: LoomingState,
        lplc2_response: LPLC2PopulationResponse,
    ) -> SpatialRiskMap:
        """
        Synthesize multi-modal visual cues into sector risk profile.
        """
        # Normalize sector nearness to [0, 1]
        nearness_risk = np.clip(sector_nearness / 3.0, 0.0, 1.0)

        # Map looming threat to corresponding sector if threat exists
        looming_risk = np.zeros(self.num_sectors, dtype=np.float32)
        if looming_state.dominant_threat is not None:
            # Azimuth in [-1, 1] mapped to sector index [0, num_sectors - 1]
            norm_az = looming_state.dominant_threat.center_x
            sec_idx = int(np.clip((norm_az + 1.0) * 0.5 * self.num_sectors, 0, self.num_sectors - 1))
            looming_risk[sec_idx] = looming_state.dominant_threat.collision_risk

            # Also bleed some risk to adjacent sectors
            if sec_idx > 0:
                looming_risk[sec_idx - 1] = max(looming_risk[sec_idx - 1], looming_risk[sec_idx] * 0.5)
            if sec_idx < self.num_sectors - 1:
                looming_risk[sec_idx + 1] = max(looming_risk[sec_idx + 1], looming_risk[sec_idx] * 0.5)

        # LPLC2 neural boost: if population is active, amplify risk in the threat direction
        neural_risk = np.zeros(self.num_sectors, dtype=np.float32)
        if lplc2_response.active_count > 0:
            norm_cx = lplc2_response.centroid_norm[0]
            sec_idx = int(np.clip((norm_cx + 1.0) * 0.5 * self.num_sectors, 0, self.num_sectors - 1))
            neural_risk[sec_idx] = min(1.0, lplc2_response.max_activation / 2.0)

        # Weighted combination
        combined = 0.45 * nearness_risk + 0.35 * looming_risk + 0.20 * neural_risk
        combined = np.clip(combined, 0.0, 1.0)

        max_risk = float(np.max(combined))
        safest_idx = int(np.argmin(combined))
        danger_idx = int(np.argmax(combined))

        if max_risk >= self.critical_threshold:
            level = "CRITICAL"
        elif max_risk >= self.warning_threshold:
            level = "WARNING"
        elif max_risk >= self.advisory_threshold:
            level = "ADVISORY"
        else:
            level = "CLEAR"

        return SpatialRiskMap(
            sector_risks=combined,
            max_risk=max_risk,
            safest_sector_idx=safest_idx,
            most_dangerous_sector_idx=danger_idx,
            overall_threat_level=level,
        )
