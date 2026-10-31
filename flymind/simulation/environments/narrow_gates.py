"""
Narrow Gates and Slit Corridor Environment.

Creates challenging narrow slit openings and gate passages to benchmark
neuromorphic Winner-Take-All (WTA) gap selection (Paper 2 / Schoepe et al.).
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple


class NarrowGatesEnvironment:
    """
    Corridor featuring a sequence of transverse wall barriers with narrow pass-through slit gates.
    """

    def __init__(
        self,
        corridor_length_m: float = 18.0,
        gate_spacing_m: float = 4.0,
        gate_width_m: float = 1.10,     # Tight slit width (drone width ~ 0.5 m)
        pillar_radius_m: float = 0.25,
        seed: int = 42,
        gate_spacing: float | None = None,
        slit_width: float | None = None,
    ):
        self.length = corridor_length_m
        self.spacing = gate_spacing if gate_spacing is not None else gate_spacing_m
        self.gate_width = slit_width if slit_width is not None else gate_width_m
        self.pillar_r = pillar_radius_m
        self.seed = seed

        self.obstacles: List[Dict[str, Any]] = []
        self.gate_centers: List[Tuple[float, float]] = []
        self.regenerate(self.seed)

    def regenerate(self, seed: int | None = None) -> List[Dict[str, Any]]:
        """Generates staggered gate corridors."""
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)

        self.obstacles = []
        self.gate_centers = []

        # Number of gates along X axis
        num_gates = int(self.length / self.spacing)
        corridor_half_width = 3.0

        for i in range(1, num_gates + 1):
            x_gate = i * self.spacing
            # Random lateral offset of the gate opening in [-1.5, +1.5]
            y_open = rng.uniform(-1.5, 1.5)
            self.gate_centers.append((x_gate, y_open))

            # Left barrier pillars
            y_left_posts = np.arange(y_open + self.gate_width / 2.0 + self.pillar_r, corridor_half_width, self.pillar_r * 2.2)
            for y_p in y_left_posts:
                self.obstacles.append({
                    "type": "cylinder",
                    "center": [float(x_gate), float(y_p)],
                    "radius": self.pillar_r,
                    "z_min": 0.0,
                    "z_max": 4.0,
                    "color": (50, 60, 90)
                })

            # Right barrier pillars
            y_right_posts = np.arange(y_open - self.gate_width / 2.0 - self.pillar_r, -corridor_half_width, -self.pillar_r * 2.2)
            for y_p in y_right_posts:
                self.obstacles.append({
                    "type": "cylinder",
                    "center": [float(x_gate), float(y_p)],
                    "radius": self.pillar_r,
                    "z_min": 0.0,
                    "z_max": 4.0,
                    "color": (50, 60, 90)
                })

        return self.obstacles

    def check_collision(self, drone_pos: np.ndarray, drone_radius: float = 0.35) -> bool:
        x, y, z = drone_pos[0], drone_pos[1], drone_pos[2]
        if z < 0.05 or z > 3.8:
            return True
        for obs in self.obstacles:
            ox, oy = obs["center"][0], obs["center"][1]
            dist = np.hypot(x - ox, y - oy)
            if dist <= (obs["radius"] + drone_radius):
                return True
        return False

    def check_clearance(self, drone_pos: np.ndarray) -> float:
        """Calculates minimum distance from drone center to nearest gate obstacle boundary."""
        x, y = drone_pos[0], drone_pos[1]
        min_d = float("inf")
        for obs in self.obstacles:
            ox, oy = obs["center"][0], obs["center"][1]
            edge_dist = np.hypot(x - ox, y - oy) - obs["radius"]
            if edge_dist < min_d:
                min_d = edge_dist
        return max(0.0, min_d)

    def get_obstacles(self, current_time: float = 0.0) -> List[Dict[str, Any]]:
        """Standardized interface returning obstacle list for camera rendering."""
        return self.obstacles


# Alias for consistent naming
NarrowGatesCorridor = NarrowGatesEnvironment

