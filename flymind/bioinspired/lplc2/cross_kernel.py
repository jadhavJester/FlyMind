"""
LPLC2 4-Lobed Cross-Kernel Receptive Field (Zhao et al., iScience 2023).

Neurobiology:
LPLC2 dendrites branch across the Lobula Plate into 4 distinct quadrants:
    - Quadrant 1 (Right): sensitive to rightward outward motion (+u)
    - Quadrant 2 (Left): sensitive to leftward outward motion (-u)
    - Quadrant 3 (Up): sensitive to upward outward motion (-v)
    - Quadrant 4 (Down): sensitive to downward outward motion (+v)

A feedforward inhibitory pool L0 sums motion across all 4 directions.
Individual lobes output: [L_k - alpha * L_0]_+.
Uniform translation excites only one or two lobes while strongly driving L0, effectively quenching response.
Only diverging expansion excites all 4 lobes simultaneously while escaping localized lateral inhibition.
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class LPLC2Subunit:
    """Activation state of an individual LPLC2 columnar neuron."""
    center_x: int
    center_y: int
    radius: int
    lobe_signals: np.ndarray  # Shape (4,) for [Right, Left, Up, Down]
    net_activation: float


class LPLC2ReceptiveField:
    """
    Implements a single LPLC2 dendritic arbor sampling an ommatidial sub-region.
    """
    def __init__(
        self,
        center_x: int,
        center_y: int,
        radius: int = 15,
        inhibition_alpha: float = 0.25,
        eps: float = 1e-6,
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.inhibition_alpha = inhibition_alpha
        self.eps = eps

    def compute(self, emd_signals: Dict[str, np.ndarray]) -> LPLC2Subunit:
        """
        Compute dendritic integration for this LPLC2 unit given EMD motion arrays.
        Args:
            emd_signals: Dict with 'right', 'left', 'up', 'down' 2D EMD activations.
        Returns:
            LPLC2Subunit with 4 lobe responses and multiplicative integrated output.
        """
        r = self.radius
        cx, cy = self.center_x, self.center_y
        h, w = emd_signals["right"].shape

        y_min = max(0, cy - r)
        y_max = min(h, cy + r)
        x_min = max(0, cx - r)
        x_max = min(w, cx + r)

        # Extract localized patches
        r_patch = emd_signals["right"][y_min:y_max, x_min:x_max]
        l_patch = emd_signals["left"][y_min:y_max, x_min:x_max]
        u_patch = emd_signals["up"][y_min:y_max, x_min:x_max]
        d_patch = emd_signals["down"][y_min:y_max, x_min:x_max]

        # Spatial integration over the 4 dendritic sectors (radially outward motion)
        # Right sector: x > cx
        # Left sector:  x < cx
        # Up sector:    y < cy
        # Down sector:  y > cy
        ph, pw = r_patch.shape
        mid_y, mid_x = ph // 2, pw // 2

        # Radial outward flow in each sector:
        sig_right = np.sum(r_patch[:, mid_x:]) if pw > mid_x else 0.0
        sig_left  = np.sum(l_patch[:, :mid_x]) if mid_x > 0 else 0.0
        sig_up    = np.sum(u_patch[:mid_y, :]) if mid_y > 0 else 0.0
        sig_down  = np.sum(d_patch[mid_y:, :]) if ph > mid_y else 0.0

        # Feedforward inhibitory signal L0
        l0 = (sig_right + sig_left + sig_up + sig_down) * self.inhibition_alpha

        # Rectified lobe responses
        lobe_r = max(0.0, float(sig_right - l0))
        lobe_l = max(0.0, float(sig_left - l0))
        lobe_u = max(0.0, float(sig_up - l0))
        lobe_d = max(0.0, float(sig_down - l0))

        lobes = np.array([lobe_r, lobe_l, lobe_u, lobe_d], dtype=np.float32)

        # Multiplicative integration (Zhao et al. 2023):
        # I_LPLC2 = prod_{k=1}^4 (lobe_k)
        # If any direction has zero outward motion (e.g. laminar translation), product is 0.
        # Scale with 4th root to maintain well-behaved dynamic range across different zoom levels
        prod = lobe_r * lobe_l * lobe_u * lobe_d
        net_activation = float(np.power(prod, 0.25)) if prod > 0.0 else 0.0

        return LPLC2Subunit(
            center_x=cx,
            center_y=cy,
            radius=r,
            lobe_signals=lobes,
            net_activation=net_activation,
        )
