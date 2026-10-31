"""
Fail-Safe State Definitions and Emergency Protocol.
"""
from enum import Enum, auto


class EmergencyState(Enum):
    NOMINAL = auto()             # Normal operation under AI or manual control
    ACTIVE_CLAMPING = auto()     # Setpoints clamped within velocity/attitude limits
    GEOFENCE_RETRACT = auto()    # Setpoints redirected back into safe volume
    EMERGENCY_HOVER = auto()     # Complete velocity freeze and altitude lock
    EMERGENCY_LAND = auto()      # Controlled vertical descent to ground
    EMERGENCY_DISARM = auto()    # Immediate motor cut-off (terminal fail-safe)


class EmergencyReason(Enum):
    NONE = auto()
    VELOCITY_LIMIT_EXCEEDED = auto()
    GEOFENCE_BREACH = auto()
    MINIMUM_CLEARANCE_VIOLATION = auto()
    WATCHDOG_TIMEOUT = auto()
    MANUAL_OPERATOR_ABORT = auto()
