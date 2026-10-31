"""
Procedural Obstacle Forest Environment Generator.

Generates reproducible cluttered arenas populated with cylindrical obstacles (trees/columns)
with Poisson-disk-like minimum distance spacing. Follows EVFly / DodgeDrone benchmark schemas.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple


class ObstacleForest:
    """
    Cluttered forest arena containing vertical cylindrical obstacles.
    """

    def __init__(
        self,
        arena_size: Tuple[float, float] = (16.0, 16.0),
        num_obstacles: int = 35,
        min_obstacle_distance: float = 1.4,
        obstacle_radius_range: Tuple[float, float] = (0.25, 0.45),
        seed: int = 123
    ):
        self.arena_x, self.arena_y = arena_size
        self.num_obstacles = num_obstacles
        self.min_dist = min_obstacle_distance
        self.radius_range = obstacle_radius_range
        self.seed = seed
        self.obstacles: List[Dict[str, Any]] = []
        self.regenerate(self.seed)

    def regenerate(self, seed: int | None = None) -> List[Dict[str, Any]]:
        """Generates a reproducible randomized forest layout."""
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)

        self.obstacles = []
        half_x, half_y = self.arena_x / 2.0, self.arena_y / 2.0
        # Exclude origin start area (radius 1.5 m)
        safe_radius_sq = 1.5 ** 2

        attempts = 0
        max_attempts = 2000

        while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(-half_x + 1.0, half_x - 1.0)
            y = rng.uniform(-half_y + 1.0, half_y - 1.0)

            # Avoid start position at (0, 0)
            if (x**2 + y**2) < safe_radius_sq:
                continue

            r = rng.uniform(self.radius_range[0], self.radius_range[1])

            # Check clearance against existing obstacles
            collision = False
            for obs in self.obstacles:
                ox, oy = obs["center"][0], obs["center"][1]
                dist = np.hypot(x - ox, y - oy)
                if dist < (r + obs["radius"] + self.min_dist):
                    collision = True
                    break

            if not collision:
                # Color variation (brown/gray wood tree trunks)
                c_val = rng.integers(35, 75)
                color = (c_val + rng.integers(-5, 5), c_val, c_val - rng.integers(5, 15))
                self.obstacles.append({
                    "type": "cylinder",
                    "center": [float(x), float(y)],
                    "radius": float(r),
                    "z_min": 0.0,
                    "z_max": 4.5,
                    "color": color
                })

        return self.obstacles

    def check_collision(self, drone_pos: np.ndarray, drone_radius: float = 0.35) -> bool:
        """Checks if drone position collides with any tree cylinder."""
        x, y, z = drone_pos[0], drone_pos[1], drone_pos[2]
        if z < 0.05 or z > 4.2:
            return True  # Floor / Ceiling strike

        for obs in self.obstacles:
            ox, oy = obs["center"][0], obs["center"][1]
            dist = np.hypot(x - ox, y - oy)
            if dist <= (obs["radius"] + drone_radius):
                return True
        return False

    def min_distance(self, drone_pos: np.ndarray) -> float:
        """Calculates minimum distance from drone center to nearest obstacle boundary."""
        x, y = drone_pos[0], drone_pos[1]
        min_d = float("inf")
        for obs in self.obstacles:
            ox, oy = obs["center"][0], obs["center"][1]
            edge_dist = np.hypot(x - ox, y - oy) - obs["radius"]
            if edge_dist < min_d:
                min_d = edge_dist
        return max(0.0, min_d)

    def check_clearance(self, drone_pos: np.ndarray) -> float:
        """Standardized interface returning min distance to obstacle boundaries."""
        return self.min_distance(drone_pos)

    def get_obstacles(self, current_time: float = 0.0) -> List[Dict[str, Any]]:
        """Standardized interface returning obstacle list for camera rendering."""
        return self.obstacles

