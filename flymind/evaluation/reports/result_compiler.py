"""
Statistical Result Compiler and Report Generator.
Aggregates multi-seed simulation metrics, performs hypothesis tests, and outputs formatted Markdown tables and CSVs.
"""
from typing import Dict, List
import numpy as np

from ..metrics.performance_metrics import FlightEpisodeMetrics


class ResultCompiler:
    """
    Summarizes multi-run benchmark episodes.
    """
    def __init__(self, metrics_list: List[FlightEpisodeMetrics]):
        self.metrics = metrics_list

    def compile_summary_table(self) -> str:
        """Generate a GitHub-flavored Markdown comparative performance table."""
        # Group by controller and scenario
        groups: Dict[str, List[FlightEpisodeMetrics]] = {}
        for m in self.metrics:
            key = f"{m.controller_name} | {m.scenario_name}"
            groups.setdefault(key, []).append(m)

        lines = [
            "| Controller | Scenario | Runs | Success (%) | Collision (%) | Mean Clear (m) | Min Clear (m) | Latency (ms) | Saccades |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for key, runs in groups.items():
            ctrl, scen = key.split(" | ")
            n = len(runs)
            success_pct = 100.0 * sum(1 for r in runs if r.reached_goal and not r.has_collided) / n
            collision_pct = 100.0 * sum(1 for r in runs if r.has_collided) / n
            mean_clear = float(np.mean([r.mean_clearance_m for r in runs]))
            min_clear = float(np.min([r.min_clearance_m for r in runs]))
            mean_lat = float(np.mean([r.mean_computation_time_ms for r in runs]))
            mean_sacc = float(np.mean([r.saccade_count for r in runs]))

            lines.append(
                f"| **{ctrl}** | {scen} | {n} | {success_pct:.1f}% | {collision_pct:.1f}% | {mean_clear:.2f}m | {min_clear:.2f}m | {mean_lat:.1f}ms | {mean_sacc:.1f} |"
            )

        return "\n".join(lines)
