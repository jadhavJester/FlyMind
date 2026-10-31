"""
FlyMind Reflex State Machine.
Coordinates behavioural transitions between cruising, gap-seeking, evasive saccade, and hover.

Inspired by Drosophila flight control:
- Smooth optomotor steering regulates course stability.
- Fast lateral saccades (~50-100 ms) are triggered by Giant Fiber / looming circuits.
- Active braking occurs when all forward apertures are occluded.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class FlightMode(Enum):
    CRUISE = auto()           # Nominal forward transit
    GAP_SEEKING = auto()      # Steer through detected optical gap
    EVASIVE_SACCADE = auto()  # Rapid bio-inspired escape maneuver
    BRAKING_HOVER = auto()    # Emergency deceleration and hold


@dataclass
class ReflexStateTransition:
    previous_mode: FlightMode
    current_mode: FlightMode
    trigger_reason: str
    saccade_direction_sign: float  # +1.0 for right escape, -1.0 for left escape


class ReflexStateMachine:
    """
    Manages high-level navigational state transitions based on perception and neural alerts.
    """
    def __init__(
        self,
        saccade_duration_s: float = 0.40,  # ~400 ms quadrotor saccade execution
        hover_settle_s: float = 0.80,
    ):
        self.saccade_duration_s = saccade_duration_s
        self.hover_settle_s = hover_settle_s

        self.mode = FlightMode.CRUISE
        self._state_timer: float = 0.0
        self._saccade_sign: float = 1.0

    def reset(self) -> None:
        self.mode = FlightMode.CRUISE
        self._state_timer = 0.0
        self._saccade_sign = 1.0

    def update(
        self,
        dt: float,
        gf_spike: bool,
        max_risk: float,
        threat_centroid_x: float,
        gap_confidence: float,
    ) -> ReflexStateTransition:
        """
        Step the reflex state machine.
        Args:
            dt: Timestep (s).
            gf_spike: True if Giant Fiber neuron fired an escape action potential.
            max_risk: Peak collision risk [0.0, 1.0].
            threat_centroid_x: Threat horizontal azimuth [-1.0 (left), 1.0 (right)].
            gap_confidence: Confidence score of open flight corridor [0.0, 1.0].
        """
        prev_mode = self.mode
        self._state_timer += dt
        reason = "Normal flight"

        # 1. Highest priority trigger: Giant Fiber spike or extreme critical risk -> EVASIVE_SACCADE
        if (gf_spike or max_risk >= 0.88) and self.mode != FlightMode.EVASIVE_SACCADE:
            self.mode = FlightMode.EVASIVE_SACCADE
            self._state_timer = 0.0
            # Saccade away from threat centroid (contralateral bias)
            self._saccade_sign = -1.0 if threat_centroid_x >= 0.0 else 1.0
            reason = f"Giant Fiber Trigger / Critical Looming Risk ({max_risk:.2f})"

        # 2. In EVASIVE_SACCADE: execute for saccade_duration_s, then transition to BRAKING_HOVER
        elif self.mode == FlightMode.EVASIVE_SACCADE:
            if self._state_timer >= self.saccade_duration_s:
                self.mode = FlightMode.BRAKING_HOVER
                self._state_timer = 0.0
                reason = "Saccade completed -> Stabilizing hover"
            else:
                reason = f"Executing evasive saccade ({self._state_timer:.2f}s)"

        # 3. In BRAKING_HOVER: wait for motion to stabilize, then re-assess
        elif self.mode == FlightMode.BRAKING_HOVER:
            if self._state_timer >= self.hover_settle_s:
                if max_risk < 0.40:
                    self.mode = FlightMode.CRUISE
                    reason = "Environment clear -> Resuming cruise"
                else:
                    self.mode = FlightMode.GAP_SEEKING
                    reason = "Obstacles remain -> Engaging gap-seeking"
            else:
                reason = f"Stabilizing hover ({self._state_timer:.2f}s)"

        # 4. In GAP_SEEKING: steer toward open sectors
        elif self.mode == FlightMode.GAP_SEEKING:
            if max_risk < 0.25 and gap_confidence > 0.70:
                self.mode = FlightMode.CRUISE
                reason = "Aperture cleared -> Resuming cruise"
            elif max_risk >= 0.75 and gap_confidence < 0.20:
                # No clear gap and high risk -> Brake
                self.mode = FlightMode.BRAKING_HOVER
                self._state_timer = 0.0
                reason = "No viable opening -> Emergency braking"
            else:
                reason = f"Navigating gap (Risk: {max_risk:.2f})"

        # 5. In CRUISE: check for obstacles entering advisory/warning zone
        elif self.mode == FlightMode.CRUISE:
            if max_risk >= 0.35:
                self.mode = FlightMode.GAP_SEEKING
                reason = f"Obstacle advisory detected (Risk: {max_risk:.2f})"

        return ReflexStateTransition(
            previous_mode=prev_mode,
            current_mode=self.mode,
            trigger_reason=reason,
            saccade_direction_sign=self._saccade_sign,
        )
