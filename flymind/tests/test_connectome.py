"""
Unit tests for MaleCNS visual-motor connectome pathway model.
"""
import numpy as np
from ..bioinspired.connectome.malecns_pathway import MaleCNSVisualMotorPathway


def test_malecns_connectome_quiescence():
    pathway = MaleCNSVisualMotorPathway(num_azimuth_sectors=8)
    lplc2_zero = np.zeros(16, dtype=np.float32)
    flow_zero = np.zeros(8, dtype=np.float32)

    state = pathway.step(lplc2_zero, flow_zero)
    # When no threat, Giant Fiber (DNp01) must stay quiet
    assert state.dnp01_giant_fiber < 0.1
    # Throttle should be at nominal cruise (~0.5)
    assert 0.4 <= state.dng02_stroke_amplitude <= 0.6
    # Yaw steering should be neutral (~0.0)
    assert abs(state.dnp02_yaw_steering) < 0.1


def test_malecns_connectome_looming_escape():
    pathway = MaleCNSVisualMotorPathway(num_azimuth_sectors=8)
    # Simulate bilateral looming expansion
    lplc2_looming = np.ones(16, dtype=np.float32) * 0.9
    flow_high = np.ones(8, dtype=np.float32) * 2.5

    for _ in range(5):
        state = pathway.step(lplc2_looming, flow_high)

    # Giant Fiber (DNp01) should charge and trigger escape
    assert state.dnp01_giant_fiber > 0.4
    assert state.active_visual_units > 10


def test_malecns_connectome_contralateral_steering():
    pathway = MaleCNSVisualMotorPathway(num_azimuth_sectors=8)
    # Looming hazard strictly on the LEFT visual field (sectors 0..7)
    lplc2_left_threat = np.zeros(16, dtype=np.float32)
    lplc2_left_threat[0:4] = 0.9
    flow_div = np.zeros(8, dtype=np.float32)

    for _ in range(3):
        state = pathway.step(lplc2_left_threat, flow_div)

    # Contralateral steering rule: Left threat -> steer RIGHT (+dnp02)
    assert state.dnp02_yaw_steering > 0.1
