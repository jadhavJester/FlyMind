"""
Test Navigation Modules: Reflex State Machine, Reactive Navigator, and Baselines.
"""
import numpy as np
import pytest

from flymind.navigation.safe_navigation.reflex_state_machine import ReflexStateMachine, FlightMode
from flymind.navigation.safe_navigation.reactive_navigator import ReactiveNavigator
from flymind.bioinspired.motion_detectors.wta_gap import GapDecision
from flymind.navigation.obstacle_awareness.collision_risk import SpatialRiskMap
from flymind.baselines.opencv_edge_detector import CannyEdgeBaseline
from flymind.baselines.monocular_depth_net import MonocularDepthBaseline


def test_reflex_state_machine_saccade_transition():
    sm = ReflexStateMachine()
    assert sm.mode == FlightMode.CRUISE

    # Normal step
    t1 = sm.update(dt=0.02, gf_spike=False, max_risk=0.1, threat_centroid_x=0.0, gap_confidence=0.9)
    assert sm.mode == FlightMode.CRUISE

    # Giant Fiber spike triggers evasive saccade immediately
    t2 = sm.update(dt=0.02, gf_spike=True, max_risk=0.9, threat_centroid_x=0.5, gap_confidence=0.1)
    assert sm.mode == FlightMode.EVASIVE_SACCADE
    assert t2.saccade_direction_sign == -1.0  # escapes contralaterally away from threat at +0.5


def test_reactive_navigator_setpoints():
    nav = ReactiveNavigator()
    risk_map = SpatialRiskMap(
        sector_risks=np.zeros(8, dtype=np.float32),
        max_risk=0.1,
        safest_sector_idx=4,
        most_dangerous_sector_idx=0,
        overall_threat_level="CLEAR",
    )
    gap_dec = GapDecision(
        best_sector_idx=4,
        target_azimuth_rad=0.0,
        sector_confidence=0.8,
        inhibited_energies=np.ones(8),
        raw_energies=np.zeros(8),
    )
    sp, _ = nav.compute_setpoints(
        dt=0.02,
        gf_spike=False,
        risk_map=risk_map,
        gap_decision=gap_dec,
        threat_centroid_x=0.0,
        current_altitude=1.5,
    )
    assert sp.vx_body > 0.0
    assert sp.mode == FlightMode.CRUISE


def test_baselines():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    canny = CannyEdgeBaseline()
    sp_c = canny.compute_setpoint(img, current_altitude=1.5)
    assert sp_c.vx_body > 0.0

    depth = MonocularDepthBaseline()
    sp_d = depth.compute_setpoint(img, current_altitude=1.5)
    assert sp_d.vx_body > 0.0
