"""
Obstacle Awareness Subpackage
Estimates collision risk, spatial obstacle margins, and time-to-contact profiles.
"""
from .collision_risk import CollisionRiskAssessor, SpatialRiskMap

__all__ = ["CollisionRiskAssessor", "SpatialRiskMap"]
