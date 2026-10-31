"""
Independent Safety Supervisor with Final Setpoint Authority.
Enforces physical flight envelopes, obstacle keep-out zones, geofencing, and heartbeat watchdogs.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ..emergency_states.fail_safe_states import EmergencyReason, EmergencyState
from ...navigation.safe_navigation.reactive_navigator import FlightSetpoint


@dataclass
class SafetyIntervention:
    """Detailed record of supervisor action."""
    original_setpoint: FlightSetpoint
    enforced_setpoint: FlightSetpoint
    safety_state: EmergencyState
    reason: EmergencyReason
    is_overridden: bool
    diagnostic_message: str


class SafetySupervisor:
    """
    Independent monitor verifying all autonomous drone setpoints before transmission to hardware.
    """
    def __init__(
        self,
        max_speed_m_s: float = 3.0,
        max_yaw_rate_rad_s: float = 2.5,
        min_clearance_m: float = 0.35,
        geofence_x: Tuple[float, float] = (-25.0, 25.0),
        geofence_y: Tuple[float, float] = (-25.0, 25.0),
        geofence_z: Tuple[float, float] = (0.2, 8.0),
        watchdog_timeout_s: float = 0.20,
    ):
        self.max_speed_m_s = max_speed_m_s
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self.min_clearance_m = min_clearance_m
        self.geofence_x = geofence_x
        self.geofence_y = geofence_y
        self.geofence_z = geofence_z
        self.watchdog_timeout_s = watchdog_timeout_s

        self.last_heartbeat_time: float = 0.0
        self.current_state = EmergencyState.NOMINAL

    def heartbeat(self, current_time: float) -> None:
        """Ping watchdog to indicate perception/control loop is alive."""
        self.last_heartbeat_time = current_time

    def verify_and_enforce(
        self,
        setpoint: FlightSetpoint,
        current_position: np.ndarray,
        estimated_clearance_m: float,
        current_time: float,
    ) -> SafetyIntervention:
        """
        Validate setpoint against hard flight safety constraints.
        Args:
            setpoint: Proposed FlightSetpoint from AI navigation.
            current_position: 3D position [x, y, z] in world frame.
            estimated_clearance_m: Distance to closest obstacle in meters.
            current_time: Current simulation/system timestamp in seconds.
        Returns:
            SafetyIntervention with either approved, clamped, or emergency-overridden setpoints.
        """
        px, py, pz = current_position

        # 1. Watchdog check
        if self.last_heartbeat_time > 0 and (current_time - self.last_heartbeat_time > self.watchdog_timeout_s):
            self.current_state = EmergencyState.EMERGENCY_HOVER
            hover_sp = FlightSetpoint(vx_body=0.0, vy_body=0.0, vz_body=0.0, yaw_rate=0.0, mode=setpoint.mode)
            return SafetyIntervention(
                original_setpoint=setpoint,
                enforced_setpoint=hover_sp,
                safety_state=self.current_state,
                reason=EmergencyReason.WATCHDOG_TIMEOUT,
                is_overridden=True,
                diagnostic_message=f"Watchdog timeout: {current_time - self.last_heartbeat_time:.3f}s elapsed",
            )

        # 2. Minimum clearance check
        if estimated_clearance_m < self.min_clearance_m:
            self.current_state = EmergencyState.EMERGENCY_HOVER
            hover_sp = FlightSetpoint(vx_body=0.0, vy_body=0.0, vz_body=0.0, yaw_rate=0.0, mode=setpoint.mode)
            return SafetyIntervention(
                original_setpoint=setpoint,
                enforced_setpoint=hover_sp,
                safety_state=self.current_state,
                reason=EmergencyReason.MINIMUM_CLEARANCE_VIOLATION,
                is_overridden=True,
                diagnostic_message=f"Clearance violated: {estimated_clearance_m:.2f}m < {self.min_clearance_m}m",
            )

        # 3. Geofence boundary check
        in_bounds_x = self.geofence_x[0] <= px <= self.geofence_x[1]
        in_bounds_y = self.geofence_y[0] <= py <= self.geofence_y[1]
        in_bounds_z = self.geofence_z[0] <= pz <= self.geofence_z[1]

        if not (in_bounds_x and in_bounds_y and in_bounds_z):
            self.current_state = EmergencyState.GEOFENCE_RETRACT
            # Redirect velocity back toward center of geofence
            target_cx = (self.geofence_x[0] + self.geofence_x[1]) / 2.0
            target_cy = (self.geofence_y[0] + self.geofence_y[1]) / 2.0
            dx = target_cx - px
            dy = target_cy - py
            dist = np.hypot(dx, dy) + 1e-4

            retract_sp = FlightSetpoint(
                vx_body=float(np.clip(dx / dist * 1.0, -1.0, 1.0)),
                vy_body=float(np.clip(dy / dist * 1.0, -1.0, 1.0)),
                vz_body=0.0,
                yaw_rate=0.0,
                mode=setpoint.mode,
            )
            return SafetyIntervention(
                original_setpoint=setpoint,
                enforced_setpoint=retract_sp,
                safety_state=self.current_state,
                reason=EmergencyReason.GEOFENCE_BREACH,
                is_overridden=True,
                diagnostic_message=f"Geofence breach at ({px:.1f}, {py:.1f}, {pz:.1f})",
            )

        # 4. Velocity magnitude clamping
        speed_body = np.hypot(setpoint.vx_body, setpoint.vy_body)
        clamped_vx = setpoint.vx_body
        clamped_vy = setpoint.vy_body
        clamped_vz = float(np.clip(setpoint.vz_body, -1.5, 1.5))
        clamped_yaw = float(np.clip(setpoint.yaw_rate, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s))

        was_clamped = False
        if speed_body > self.max_speed_m_s:
            scale = self.max_speed_m_s / speed_body
            clamped_vx *= scale
            clamped_vy *= scale
            was_clamped = True

        if abs(setpoint.yaw_rate) > self.max_yaw_rate_rad_s or setpoint.vz_body != clamped_vz:
            was_clamped = True

        enforced_sp = FlightSetpoint(
            vx_body=clamped_vx,
            vy_body=clamped_vy,
            vz_body=clamped_vz,
            yaw_rate=clamped_yaw,
            mode=setpoint.mode,
        )

        if was_clamped:
            self.current_state = EmergencyState.ACTIVE_CLAMPING
            return SafetyIntervention(
                original_setpoint=setpoint,
                enforced_setpoint=enforced_sp,
                safety_state=self.current_state,
                reason=EmergencyReason.VELOCITY_LIMIT_EXCEEDED,
                is_overridden=True,
                diagnostic_message="Setpoints clamped to envelope limits",
            )

        self.current_state = EmergencyState.NOMINAL
        return SafetyIntervention(
            original_setpoint=setpoint,
            enforced_setpoint=enforced_sp,
            safety_state=self.current_state,
            reason=EmergencyReason.NONE,
            is_overridden=False,
            diagnostic_message="All constraints satisfied",
        )
