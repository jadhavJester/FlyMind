"""
FlyMind Conventional Baselines Subpackage
Includes classical computer-vision edge density and lightweight monocular depth estimation for benchmarking against bio-inspired methods.
"""
from .opencv_edge_detector import CannyEdgeBaseline
from .monocular_depth_net import MonocularDepthBaseline

__all__ = ["CannyEdgeBaseline", "MonocularDepthBaseline"]
