"""
Looming Detection Subpackage
Detects imminent collisions via optical expansion rates, time-to-contact, and IMU-derotated flow.
"""
from .angular_expansion import LoomingDetector, LoomingState
from .imu_derotation import IMUFlowDerotator

__all__ = ["LoomingDetector", "LoomingState", "IMUFlowDerotator"]
