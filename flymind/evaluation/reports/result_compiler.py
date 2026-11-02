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
        # Check if multiple lighting modes exist
        has_multi_lighting = len(set(getattr(m, "lighting", "day") for m in self.metrics)) > 1

        groups: Dict[str, List[FlightEpisodeMetrics]] = {}
        for m in self.metrics:
            lit = getattr(m, "lighting", "day")
            key = f"{m.controller_name} | {m.scenario_name}" + (f" | {lit}" if has_multi_lighting else "")
            groups.setdefault(key, []).append(m)

        if has_multi_lighting:
            lines = [
                "| Controller | Scenario | Lighting | Runs | Success (%) | Collision (%) | Mean Clear (m) | Min Clear (m) | Latency (ms) | Saccades |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        else:
            lines = [
                "| Controller | Scenario | Runs | Success (%) | Collision (%) | Mean Clear (m) | Min Clear (m) | Latency (ms) | Saccades |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]

        for key, runs in groups.items():
            parts = key.split(" | ")
            ctrl, scen = parts[0], parts[1]
            lit = parts[2] if len(parts) > 2 else ""
            n = len(runs)
            success_pct = 100.0 * sum(1 for r in runs if r.reached_goal and not r.has_collided) / n
            collision_pct = 100.0 * sum(1 for r in runs if r.has_collided) / n
            mean_clear = float(np.mean([r.mean_clearance_m for r in runs]))
            min_clear = float(np.min([r.min_clearance_m for r in runs]))
            mean_lat = float(np.mean([r.mean_computation_time_ms for r in runs]))
            mean_sacc = float(np.mean([r.saccade_count for r in runs]))

            if has_multi_lighting:
                lines.append(
                    f"| **{ctrl}** | {scen} | {lit} | {n} | {success_pct:.1f}% | {collision_pct:.1f}% | {mean_clear:.2f}m | {min_clear:.2f}m | {mean_lat:.1f}ms | {mean_sacc:.1f} |"
                )
            else:
                lines.append(
                    f"| **{ctrl}** | {scen} | {n} | {success_pct:.1f}% | {collision_pct:.1f}% | {mean_clear:.2f}m | {min_clear:.2f}m | {mean_lat:.1f}ms | {mean_sacc:.1f} |"
                )

        return "\n".join(lines)

    def compile_lighting_table(self) -> str:
        """Generate a degradation table comparing controllers across lighting levels."""
        controllers = sorted(list(set(m.controller_name for m in self.metrics)))
        lightings = ["day", "overcast", "dim", "dark"]

        lines = [
            "| Controller | Day Success | Overcast Success | Dim Success | Dark Success | Mean Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for ctrl in controllers:
            row = [f"**{ctrl}**"]
            latencies = []
            for lit in lightings:
                runs = [m for m in self.metrics if m.controller_name == ctrl and getattr(m, "lighting", "day") == lit]
                if runs:
                    succ = 100.0 * sum(1 for r in runs if r.reached_goal and not r.has_collided) / len(runs)
                    row.append(f"{succ:.1f}%")
                    latencies.extend([r.mean_computation_time_ms for r in runs])
                else:
                    row.append("N/A")
            mean_lat = f"{np.mean(latencies):.1f}ms" if latencies else "N/A"
            row.append(mean_lat)
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
