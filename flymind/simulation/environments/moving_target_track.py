"""
10-Meter Flight Track with Dynamic Moving Target Environment.
Simulates a structured 10-12m aerial race track / corridor with guide pylons
and a dynamic moving target obstacle crossing and patrolling the flight corridor.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple


class MovingTarget:
    """
    A dynamic 3D moving target operating within the 10m track corridor.
    Can operate in:
      - 'crossing': lateral harmonic patrol across the track corridor (cuts across flight path)
      - 'patrol': longitudinal track navigation ahead of the drone with evasive weave
    """
    def __init__(
        self,
        initial_pos: Tuple[float, float, float] = (5.5, 0.0, 1.5),
        radius: float = 0.45,
        speed: float = 1.2,
        mode: str = "crossing",
    ):
        self.initial_pos = np.array(initial_pos, dtype=np.float64)
        self.current_pos = self.initial_pos.copy()
        self.radius = radius
        self.speed = speed
        self.mode = mode
        self.velocity = np.zeros(3, dtype=np.float64)

    def reset(self):
        self.current_pos = self.initial_pos.copy()
        self.velocity = np.zeros(3, dtype=np.float64)

    def step(self, current_time: float, dt: float = 0.02) -> np.ndarray:
        if self.mode == "crossing":
            # Lateral sinusoidal sweep across the track corridor at x ~ 5.5m
            y = 1.5 * np.sin(self.speed * current_time)
            x = self.initial_pos[0] + 0.3 * np.cos(0.5 * self.speed * current_time)
            z = self.initial_pos[2] + 0.25 * np.sin(2.0 * self.speed * current_time)
            new_pos = np.array([x, y, z], dtype=np.float64)
            self.velocity = (new_pos - self.current_pos) / max(1e-4, dt)
            self.current_pos = new_pos

        elif self.mode == "patrol":
            # Longitudinal forward flight ahead of the UAV with evasive weaving
            x = 3.5 + (self.speed * 0.8 * current_time) % 8.0
            y = 0.9 * np.sin(1.5 * current_time)
            z = 1.5 + 0.2 * np.cos(1.8 * current_time)
            new_pos = np.array([x, y, z], dtype=np.float64)
            self.velocity = (new_pos - self.current_pos) / max(1e-4, dt)
            self.current_pos = new_pos

        return self.current_pos.copy()

    def get_obstacle_spec(self) -> Dict[str, Any]:
        """Returns visual obstacle specification for SyntheticCamera."""
        return {
            "type": "sphere",
            "center": [float(self.current_pos[0]), float(self.current_pos[1]), float(self.current_pos[2])],
            "radius": float(self.radius),
            "color": (220, 40, 40),  # Bright red target sphere with high contrast
            "name": "moving_target",
        }


class MovingTargetTrack:
    """
    10-Meter Track Arena with Corridor Pylons and a Dynamic Moving Target.
    Total track length: 12.0 meters (covering 10+ meters of continuous flight).
    Corridor width: 3.2 meters (-1.6m to +1.6m lateral bounds).
    """
    def __init__(
        self,
        track_length_m: float = 12.0,
        corridor_width_m: float = 3.2,
        target_mode: str = "crossing",
        target_speed: float = 1.5,
    ):
        self.track_length = track_length_m
        self.corridor_width = corridor_width_m
        self.target = MovingTarget(
            initial_pos=(5.5, 0.0, 1.5),
            radius=0.45,
            speed=target_speed,
            mode=target_mode,
        )
        self.last_time = 0.0

        # Build track corridor guide pylons
        # Pairs of left/right pylons every 2m from x = 1.5m to x = 11.5m
        self.static_pylons: List[Dict[str, Any]] = []
        half_w = self.corridor_width / 2.0

        xs = np.arange(1.5, self.track_length, 2.0)
        for idx, x_p in enumerate(xs):
            # Alternating track colors (orange/white race markers)
            col = (230, 110, 30) if idx % 2 == 0 else (210, 210, 210)

            # Left pylon
            self.static_pylons.append({
                "type": "cylinder",
                "center": [float(x_p), float(-half_w)],
                "radius": 0.22,
                "z_min": 0.0,
                "z_max": 3.0,
                "color": col,
            })
            # Right pylon
            self.static_pylons.append({
                "type": "cylinder",
                "center": [float(x_p), float(half_w)],
                "radius": 0.22,
                "z_min": 0.0,
                "z_max": 3.0,
                "color": col,
            })

        # Add finish line gate at x = track_length - 0.5m
        x_finish = self.track_length - 0.5
        self.static_pylons.append({
            "type": "cylinder",
            "center": [float(x_finish), -half_w],
            "radius": 0.28,
            "z_min": 0.0,
            "z_max": 3.5,
            "color": (40, 200, 40),  # Green finish post left
        })
        self.static_pylons.append({
            "type": "cylinder",
            "center": [float(x_finish), half_w],
            "radius": 0.28,
            "z_min": 0.0,
            "z_max": 3.5,
            "color": (40, 200, 40),  # Green finish post right
        })

    def get_obstacles(self, current_time: float = 0.0) -> List[Dict[str, Any]]:
        dt = max(0.001, current_time - self.last_time) if current_time > self.last_time else 0.02
        self.target.step(current_time, dt=dt)
        self.last_time = current_time

        # Combine static track boundaries and dynamic moving target
        all_obs = list(self.static_pylons)
        all_obs.append(self.target.get_obstacle_spec())
        return all_obs

    def check_clearance(self, drone_pos: np.ndarray) -> float:
        """Computes minimum Euclidean clearance to any pylon or the moving target."""
        min_dist = float("inf")

        # 1. Clearance to moving target
        target_diff = drone_pos - self.target.current_pos
        dist_target = float(np.linalg.norm(target_diff)) - self.target.radius
        min_dist = min(min_dist, dist_target)

        # 2. Clearance to static pylons (cylinders)
        p_xy = drone_pos[:2]
        for pylon in self.static_pylons:
            c = np.array(pylon["center"][:2], dtype=np.float64)
            r = pylon["radius"]
            d = float(np.linalg.norm(p_xy - c)) - r
            min_dist = min(min_dist, d)

        return max(0.0, min_dist)

    def check_collision(self, drone_pos: np.ndarray, drone_radius: float = 0.35) -> bool:
        return self.check_clearance(drone_pos) <= drone_radius
