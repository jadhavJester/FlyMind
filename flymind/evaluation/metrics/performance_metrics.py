"""
Flight Performance Metrics and Statistical Tracking.
Quantifies navigation safety, clearance, agility, energy, and computational efficiency.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class FlightEpisodeMetrics:
    """Metrics collected over a single flight episode."""
    controller_name: str
    scenario_name: str
    seed: int
    duration_s: float
    path_length_m: float
    mean_speed_m_s: float
    min_clearance_m: float
    mean_clearance_m: float
    has_collided: bool
    reached_goal: bool
    saccade_count: int
    mean_computation_time_ms: float
    time_to_first_reaction_s: Optional[float] = None
    trajectory_3d: List[List[float]] = field(default_factory=list)  # [[x, y, z, yaw], ...]
    obstacles: List[dict] = field(default_factory=list)


class MetricsCollector:
    """
    Accumulates timestep telemetry and computes comprehensive flight summary metrics.
    """
    def __init__(self, controller_name: str, scenario_name: str, seed: int):
        self.controller_name = controller_name
        self.scenario_name = scenario_name
        self.seed = seed

        self.positions: List[np.ndarray] = []
        self.trajectory_3d: List[List[float]] = []
        self.obstacles: List[dict] = []
        self.speeds: List[float] = []
        self.clearances: List[float] = []
        self.step_times_ms: List[float] = []
        self.saccade_events: int = 0
        self.has_collided: bool = False
        self.reached_goal: bool = False
        self.first_reaction_time: Optional[float] = None
        self.total_time_s: float = 0.0

    def record_step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        clearance_m: float,
        step_time_ms: float,
        is_saccading: bool,
        current_time_s: float,
        yaw: float = 0.0,
    ) -> None:
        """Log one simulation step."""
        pos = np.array(position, copy=True)
        self.positions.append(pos)
        self.trajectory_3d.append([float(pos[0]), float(pos[1]), float(pos[2]), float(yaw)])
        speed = float(np.linalg.norm(velocity[:2]))  # planar ground speed
        self.speeds.append(speed)
        self.clearances.append(float(clearance_m))
        self.step_times_ms.append(float(step_time_ms))
        self.total_time_s = current_time_s


        if is_saccading and self.first_reaction_time is None:
            self.first_reaction_time = current_time_s

    def register_saccade(self) -> None:
        self.saccade_events += 1

    def register_collision(self) -> None:
        self.has_collided = True

    def register_goal_reached(self) -> None:
        self.reached_goal = True

    def compute_summary(self) -> FlightEpisodeMetrics:
        """Calculate final episode statistics."""
        if len(self.positions) > 1:
            diffs = np.diff(np.array(self.positions), axis=0)
            path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            path_len = 0.0

        mean_speed = float(np.mean(self.speeds)) if self.speeds else 0.0
        min_clear = float(np.min(self.clearances)) if self.clearances else 0.0
        mean_clear = float(np.mean(self.clearances)) if self.clearances else 0.0
        mean_comp = float(np.mean(self.step_times_ms)) if self.step_times_ms else 0.0

        return FlightEpisodeMetrics(
            controller_name=self.controller_name,
            scenario_name=self.scenario_name,
            seed=self.seed,
            duration_s=self.total_time_s,
            path_length_m=path_len,
            mean_speed_m_s=mean_speed,
            min_clearance_m=min_clear,
            mean_clearance_m=mean_clear,
            has_collided=self.has_collided,
            reached_goal=self.reached_goal,
            saccade_count=self.saccade_events,
            mean_computation_time_ms=mean_comp,
            time_to_first_reaction_s=self.first_reaction_time,
            trajectory_3d=self.trajectory_3d,
            obstacles=self.obstacles,
        )

