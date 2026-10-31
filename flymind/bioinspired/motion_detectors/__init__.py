"""
Motion Detectors and Gap Selection Subpackage.
Implements biological Winner-Take-All lateral inhibition networks for aperture selection (Schoepe et al., 2024).
"""
from .wta_gap import WinnerTakeAllGapSelector, GapDecision

__all__ = ["WinnerTakeAllGapSelector", "GapDecision"]
