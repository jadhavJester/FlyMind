"""
LPLC2 Population Array and Retinotopic Coding.
Tiles the compound eye visual field with a grid of LPLC2 units, outputting the active population size N_act(t) and spatial threat centroid.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .cross_kernel import LPLC2ReceptiveField, LPLC2Subunit


@dataclass
class LPLC2PopulationResponse:
    """Population readout from the LPLC2 neural sheet."""
    active_count: int               # N_act(t): number of neurons exceeding firing threshold
    total_population_size: int      # Total units in sheet
    max_activation: float           # Strongest individual unit activation
    mean_activation: float          # Average activation across all units
    centroid_norm: Tuple[float, float]  # Normalized azimuth/elevation of looming epicenter [-1.0, 1.0]
    activation_map: np.ndarray      # 2D grid of unit activations (grid_rows, grid_cols)


class LPLC2Population:
    """
    Grid array of LPLC2 receptive fields covering the visual field.
    """
    def __init__(
        self,
        image_width: int = 160,
        image_height: int = 120,
        grid_cols: int = 8,
        grid_rows: int = 6,
        activation_threshold: float = 0.5,
        rf_radius: int = 16,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.activation_threshold = activation_threshold

        self.units: List[LPLC2ReceptiveField] = []
        step_x = image_width // (grid_cols + 1)
        step_y = image_height // (grid_rows + 1)

        for r in range(grid_rows):
            cy = (r + 1) * step_y
            for c in range(grid_cols):
                cx = (c + 1) * step_x
                self.units.append(
                    LPLC2ReceptiveField(
                        center_x=cx,
                        center_y=cy,
                        radius=rf_radius,
                    )
                )

    def process(self, emd_signals: Dict[str, np.ndarray]) -> LPLC2PopulationResponse:
        """
        Process EMD motion field across all LPLC2 receptive fields.
        Args:
            emd_signals: Dict of 2D EMD activations ('right', 'left', 'up', 'down').
        Returns:
            LPLC2PopulationResponse summary.
        """
        activations = np.zeros(len(self.units), dtype=np.float32)
        active_indices = []

        for idx, unit in enumerate(self.units):
            sub = unit.compute(emd_signals)
            activations[idx] = sub.net_activation
            if sub.net_activation >= self.activation_threshold:
                active_indices.append(idx)

        active_count = len(active_indices)
        max_act = float(np.max(activations)) if len(activations) > 0 else 0.0
        mean_act = float(np.mean(activations)) if len(activations) > 0 else 0.0

        # Compute centroid of threat in normalized coords [-1, 1]
        if active_count > 0:
            active_acts = activations[active_indices]
            total_act = np.sum(active_acts) + 1e-6
            weighted_cx = sum(self.units[i].center_x * activations[i] for i in active_indices) / total_act
            weighted_cy = sum(self.units[i].center_y * activations[i] for i in active_indices) / total_act

            norm_cx = (weighted_cx - self.image_width / 2.0) / (self.image_width / 2.0)
            norm_cy = (weighted_cy - self.image_height / 2.0) / (self.image_height / 2.0)
        else:
            norm_cx, norm_cy = 0.0, 0.0

        act_map = activations.reshape((self.grid_rows, self.grid_cols))

        return LPLC2PopulationResponse(
            active_count=active_count,
            total_population_size=len(self.units),
            max_activation=max_act,
            mean_activation=mean_act,
            centroid_norm=(norm_cx, norm_cy),
            activation_map=act_map,
        )
