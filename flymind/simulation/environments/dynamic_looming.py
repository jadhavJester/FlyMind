"""
Dynamic Looming Hazard Environment.

Spawns dynamic moving obstacles on direct collision courses with the UAV
to test LPLC2/Giant Fiber looming detection and sub-10ms escape reflexes (Paper 1).
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple


class DynamicLoomingHazard:
    """
    Simulates approaching spherical / box obstacles moving directly toward the UAV
    with specified size-to-velocity ratio L/v.
    """

    def __init__(
        self,
        spawn_pos: Tuple[float, float, float] = (8.0, 0.0, 1.2),
        target_pos: Tuple[float, float, float] = (0.0, 0.0, 1.2),
        approach_speed_mps: float = 3.0,
        hazard_radius_m: float = 0.45
    ):
        self.spawn_pos = np.array(spawn_pos, dtype=np.float64)
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.speed = approach_speed_mps
        self.radius = hazard_radius_m

        # Compute velocity unit vector
        dir_vec = self.target_pos - self.spawn_pos
        self.direction = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
        self.velocity = self.direction * self.speed

        self.current_pos = self.spawn_pos.copy()
        self.active = True

    def reset(self, spawn_pos: Tuple[float, float, float] | None = None, speed: float | None = None):
        if spawn_pos is not None:
            self.spawn_pos = np.array(spawn_pos, dtype=np.float64)
        if speed is not None:
            self.speed = speed
        self.current_pos = self.spawn_pos.copy()
        dir_vec = self.target_pos - self.current_pos
        self.direction = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
        self.velocity = self.direction * self.speed
        self.active = True

    def step(self, dt: float = 0.01) -> np.ndarray:
        """Propagates obstacle along collision course."""
        if self.active:
            self.current_pos += self.velocity * dt
            # If passes target by 2m, deactivate
            if np.dot(self.current_pos - self.target_pos, self.direction) > 2.0:
                self.active = False
        return self.current_pos.copy()

    def get_obstacle_spec(self) -> Dict[str, Any]:
        """Returns standard renderable obstacle dictionary."""
        return {
            "type": "sphere",
            "center": [float(self.current_pos[0]), float(self.current_pos[1]), float(self.current_pos[2])],
            "radius": float(self.radius),
            "color": (20, 20, 25) # Dark looming object on light background (Paper 1 stimulus)
        }

    @property
    def l_over_v_s(self) -> float:
        """Returns half-size to velocity ratio L/v in seconds."""
        return float(self.radius / max(0.1, self.speed))


class DynamicLoomingArena:
    """
    Simulation arena containing dynamic looming hazards approaching the drone.
    """
    def __init__(
        self,
        approach_speed: float = 2.5,
        initial_distance: float = 15.0,
        hazard_radius: float = 0.50,
    ):
        self.hazard = DynamicLoomingHazard(
            spawn_pos=(initial_distance, 0.0, 1.5),
            target_pos=(0.0, 0.0, 1.5),
            approach_speed_mps=approach_speed,
            hazard_radius_m=hazard_radius,
        )
        self.last_time: float = 0.0

    def get_obstacles(self, current_time: float = 0.0) -> List[Dict[str, Any]]:
        dt = max(0.001, current_time - self.last_time) if current_time > self.last_time else 0.01
        self.hazard.step(dt)
        self.last_time = current_time
        return [self.hazard.get_obstacle_spec()]

    def check_clearance(self, drone_pos: np.ndarray) -> float:
        diff = drone_pos - self.hazard.current_pos
        dist = float(np.linalg.norm(diff)) - self.hazard.radius
        return max(0.0, dist)

    def check_collision(self, drone_pos: np.ndarray, drone_radius: float = 0.35) -> bool:
        return self.check_clearance(drone_pos) <= drone_radius

