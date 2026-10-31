"""
Nearness Estimation Subpackage
Extracts depth-reciprocal obstacle proximity from optical flow divergence fields.
"""
from .flow_divergence import FlowDivergenceNearness

__all__ = ["FlowDivergenceNearness"]
