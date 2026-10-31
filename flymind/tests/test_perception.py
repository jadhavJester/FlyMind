"""
Test Perception Modules: Optical Flow, HR-EMD, Looming Detection, IMU Derotation, Flow Divergence Nearness, DVS.
"""
import numpy as np
import pytest

from flymind.perception.optical_flow.lucas_kanade import LucasKanadeFlow
from flymind.perception.optical_flow.farneback_dense import FarnebackDenseFlow
from flymind.perception.elementary_motion_detection.hr_emd import HRElementaryMotionDetector
from flymind.perception.looming_detection.angular_expansion import LoomingDetector
from flymind.perception.looming_detection.imu_derotation import IMUFlowDerotator
from flymind.perception.nearness_estimation.flow_divergence import FlowDivergenceNearness
from flymind.perception.event_vision.event_emulator import DVSEventEmulator


def test_farneback_dense_flow():
    flow = FarnebackDenseFlow()
    f1 = np.zeros((120, 160), dtype=np.uint8)
    f2 = np.zeros((120, 160), dtype=np.uint8)
    f1[40:80, 40:80] = 255
    f2[40:80, 45:85] = 255  # shifted right by 5 pixels

    u1, v1 = flow.compute(f1)
    u2, v2 = flow.compute(f2)
    assert u2.shape == (120, 160)
    assert v2.shape == (120, 160)
    # Average horizontal flow inside the moving region should be positive (rightward)
    assert np.mean(u2[40:80, 45:85]) > 0.0


def test_hr_emd():
    emd = HRElementaryMotionDetector(dt=0.02)
    f1 = np.zeros((60, 80), dtype=np.uint8)
    f2 = np.zeros((60, 80), dtype=np.uint8)
    f1[:, 20:30] = 200
    f2[:, 22:32] = 200  # rightward motion

    r1 = emd.process_frame(f1)
    r2 = emd.process_frame(f2)
    assert "flow_u" in r2
    assert "flow_v" in r2
    assert np.mean(r2["right"]) >= 0.0


def test_imu_derotation():
    derotator = IMUFlowDerotator(width=160, height=120, fov_deg=90.0)
    omega = np.array([0.0, 0.5, 0.0])  # yaw rate of 0.5 rad/s
    u_rot, v_rot = derotator.compute_rotational_flow(omega, dt=0.02)

    assert u_rot.shape == (120, 160)
    assert v_rot.shape == (120, 160)
    # Pure yaw produces negative horizontal flow across image center
    assert u_rot[60, 80] < 0.0

    # Test derotate cancels pure rotation
    u_trans, v_trans = derotator.derotate(u_rot, v_rot, omega, dt=0.02)
    assert np.allclose(u_trans, 0.0, atol=1e-5)
    assert np.allclose(v_trans, 0.0, atol=1e-5)


def test_flow_divergence_nearness():
    nearness_est = FlowDivergenceNearness()
    # Construct diverging flow field: u(x, y) = x, v(x, y) = y
    xs = np.linspace(-10, 10, 160)
    ys = np.linspace(-10, 10, 120)
    xx, yy = np.meshgrid(xs, ys)

    near_map = nearness_est.compute(xx.astype(np.float32), yy.astype(np.float32), forward_speed=1.5, dt=0.02)
    assert near_map.shape == (120, 160)
    assert np.all(near_map >= 0.0)

    sectors = nearness_est.compute_sector_nearness(near_map, num_sectors=8)
    assert len(sectors) == 8


def test_dvs_event_emulator():
    dvs = DVSEventEmulator(contrast_threshold=0.2)
    f1 = np.ones((60, 80), dtype=np.uint8) * 100
    f2 = np.ones((60, 80), dtype=np.uint8) * 100
    f2[20:40, 20:40] = 200  # sudden brightness jump

    pkt1 = dvs.generate_events(f1, 0.0)
    pkt2 = dvs.generate_events(f2, 0.02)
    assert pkt2.event_count > 0
    assert len(pkt2.xs) == pkt2.event_count
    assert pkt2.density_map.shape == (60, 80)
