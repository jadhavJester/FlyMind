"""
Test Safety Subsystem: Independent SafetySupervisor, Geofencing, Min Clearance, Watchdog.
"""
import numpy as np
import pytest

from flymind.safety.safety_supervisor.supervisor import SafetySupervisor
from flymind.safety.emergency_states.fail_safe_states import EmergencyState, EmergencyReason
from flymind.navigation.safe_navigation.reactive_navigator import FlightSetpoint
from flymind.navigation.safe_navigation.reflex_state_machine import FlightMode


def test_safety_supervisor_nominal():
    sup = SafetySupervisor()
    sup.heartbeat(1.0)
    sp = FlightSetpoint(vx_body=1.5, vy_body=0.0, vz_body=0.0, yaw_rate=0.0, mode=FlightMode.CRUISE)
    res = sup.verify_and_enforce(
        setpoint=sp,
        current_position=np.array([0.0, 0.0, 1.5]),
        estimated_clearance_m=2.0,
        current_time=1.05,
    )
    assert not res.is_overridden
    assert res.safety_state == EmergencyState.NOMINAL


def test_safety_supervisor_clearance_override():
    sup = SafetySupervisor(min_clearance_m=0.35)
    sup.heartbeat(1.0)
    sp = FlightSetpoint(vx_body=1.5, vy_body=0.0, vz_body=0.0, yaw_rate=0.0, mode=FlightMode.CRUISE)
    # Dangerously close obstacle: 0.15 m
    res = sup.verify_and_enforce(
        setpoint=sp,
        current_position=np.array([0.0, 0.0, 1.5]),
        estimated_clearance_m=0.15,
        current_time=1.05,
    )
    assert res.is_overridden
    assert res.safety_state == EmergencyState.EMERGENCY_HOVER
    assert res.reason == EmergencyReason.MINIMUM_CLEARANCE_VIOLATION
    assert res.enforced_setpoint.vx_body == 0.0


def test_safety_supervisor_geofence_override():
    sup = SafetySupervisor(geofence_x=(-10.0, 10.0), geofence_y=(-10.0, 10.0), geofence_z=(0.2, 5.0))
    sup.heartbeat(1.0)
    sp = FlightSetpoint(vx_body=1.5, vy_body=0.0, vz_body=0.0, yaw_rate=0.0, mode=FlightMode.CRUISE)
    # Position outside geofence: x = 15.0
    res = sup.verify_and_enforce(
        setpoint=sp,
        current_position=np.array([15.0, 0.0, 1.5]),
        estimated_clearance_m=3.0,
        current_time=1.05,
    )
    assert res.is_overridden
    assert res.safety_state == EmergencyState.GEOFENCE_RETRACT
    assert res.reason == EmergencyReason.GEOFENCE_BREACH
    # Velocity should be directed back toward center (negative vx)
    assert res.enforced_setpoint.vx_body < 0.0


def test_safety_supervisor_watchdog_timeout():
    sup = SafetySupervisor(watchdog_timeout_s=0.15)
    sup.heartbeat(1.0)
    sp = FlightSetpoint(vx_body=1.5, vy_body=0.0, vz_body=0.0, yaw_rate=0.0, mode=FlightMode.CRUISE)
    # Time elapsed: 0.35s > 0.15s
    res = sup.verify_and_enforce(
        setpoint=sp,
        current_position=np.array([0.0, 0.0, 1.5]),
        estimated_clearance_m=2.0,
        current_time=1.35,
    )
    assert res.is_overridden
    assert res.safety_state == EmergencyState.EMERGENCY_HOVER
    assert res.reason == EmergencyReason.WATCHDOG_TIMEOUT
