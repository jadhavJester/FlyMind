"""
Reactive Bio-Inspired Velocity and Heading Setpoint Generator.
Generates 3D velocity setpoints [vx, vy, vz] and yaw rates r_cmd for low-level drone autopilots.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .reflex_state_machine import FlightMode, ReflexStateMachine, ReflexStateTransition
from ...bioinspired.motion_detectors.wta_gap import GapDecision
from ..obstacle_awareness.collision_risk import SpatialRiskMap


@dataclass
class FlightSetpoint:
    """Velocity and heading setpoints sent to drone attitude/velocity controller."""
    vx_body: float       # Forward velocity in body frame (m/s)
    vy_body: float       # Lateral velocity in body frame (m/s, + right, - left)
    vz_body: float       # Vertical velocity in body frame (m/s, + down or altitude rate)
    yaw_rate: float      # Desired yaw angular velocity (rad/s)
    mode: FlightMode     # Active behavioural flight mode


class ReactiveNavigator:
    """
    Bio-inspired reactive guidance controller.
    Synthesizes optical flow gap finding, looming alerts, and optomotor balance into smooth flight setpoints.
    """
    def __init__(
        self,
        nominal_cruise_speed: float = 1.5,   # m/s forward speed
        max_saccade_speed: float = 2.5,      # m/s lateral escape speed
        yaw_gain: float = 1.8,               # Proportional gain for gap steering
        max_yaw_rate: float = 2.0,           # rad/s max turn rate
        target_altitude_m: float = 1.5,
    ):
        self.nominal_cruise_speed = nominal_cruise_speed
        self.max_saccade_speed = max_saccade_speed
        self.yaw_gain = yaw_gain
        self.max_yaw_rate = max_yaw_rate
        self.target_altitude_m = target_altitude_m

        self.state_machine = ReflexStateMachine()

    def reset(self) -> None:
        self.state_machine.reset()

    def compute_setpoints(
        self,
        dt: float,
        gf_spike: bool,
        risk_map: SpatialRiskMap,
        gap_decision: GapDecision,
        threat_centroid_x: float,
        current_altitude: float,
    ) -> Tuple[FlightSetpoint, ReflexStateTransition]:
        """
        Compute body-frame velocity setpoints and yaw rate.
        """
        # Step state machine
        transition = self.state_machine.update(
            dt=dt,
            gf_spike=gf_spike,
            max_risk=risk_map.max_risk,
            threat_centroid_x=threat_centroid_x,
            gap_confidence=gap_decision.sector_confidence,
        )

        mode = self.state_machine.mode

        # Altitude hold regulation (P-controller)
        alt_error = self.target_altitude_m - current_altitude
        vz_cmd = float(np.clip(-1.0 * alt_error, -0.8, 0.8))  # standard NED frame: negative is up

        if mode == FlightMode.CRUISE:
            vx = self.nominal_cruise_speed
            vy = 0.0
            yaw_rate = 0.0

        elif mode == FlightMode.GAP_SEEKING:
            # Throttle forward speed with risk: slower in tighter passages
            speed_factor = max(0.25, 1.0 - risk_map.max_risk)
            vx = self.nominal_cruise_speed * speed_factor

            # Camera azimuth is positive rightwards (+X_cam = -Y_body)
            # Turning right in right-handed Z-up frame corresponds to negative yaw rate
            desired_yaw_rate = -self.yaw_gain * gap_decision.target_azimuth_rad
            yaw_rate = float(np.clip(desired_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

            # Optomotor lateral drift: steer laterally toward gap opening (right = -Y_body)
            vy = -float(np.clip(gap_decision.target_azimuth_rad * 0.8, -0.6, 0.6))

        elif mode == FlightMode.EVASIVE_SACCADE:
            # All-or-none high-speed lateral escape
            sign = transition.saccade_direction_sign
            vx = 0.3 * self.nominal_cruise_speed  # preserve minimal forward momentum
            # If threat is right (sign < 0), escape left (+Y_body)
            vy = -sign * self.max_saccade_speed    # intense lateral burst
            vz_cmd = -0.5                          # slight climb during escape saccade
            yaw_rate = -sign * self.max_yaw_rate   # rapid heading whip away from hazard


        elif mode == FlightMode.BRAKING_HOVER:
            # Zero body velocity
            vx = 0.0
            vy = 0.0
            yaw_rate = 0.0

        else:
            vx, vy, yaw_rate = 0.0, 0.0, 0.0

        setpoint = FlightSetpoint(
            vx_body=vx,
            vy_body=vy,
            vz_body=vz_cmd,
            yaw_rate=yaw_rate,
            mode=mode,
        )
        return setpoint, transition
