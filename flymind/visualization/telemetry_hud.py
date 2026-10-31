"""
OpenCV Telemetry HUD (Heads-Up Display) Overlay.
Renders real-time bio-inspired perception, neural signals, optical flow vectors, and flight modes directly onto the video stream.
"""
from typing import Optional, Tuple
import cv2
import numpy as np

from ..navigation.safe_navigation.reflex_state_machine import FlightMode
from ..navigation.obstacle_awareness.collision_risk import SpatialRiskMap
from ..bioinspired.neural_models.giant_fiber_lif import GFSpikeState
from ..safety.emergency_states.fail_safe_states import EmergencyState


class TelemetryHUD:
    """
    Overlays flight status, optical flow field, nearness indicators, and neural potentials.
    """
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    def render(
        self,
        base_frame: np.ndarray,
        flow_u: Optional[np.ndarray],
        flow_v: Optional[np.ndarray],
        mode: FlightMode,
        gf_state: Optional[GFSpikeState],
        risk_map: Optional[SpatialRiskMap],
        safety_state: EmergencyState,
        speed_m_s: float,
        altitude_m: float,
        yaw_deg: float,
    ) -> np.ndarray:
        """
        Produce composited telemetry frame.
        """
        # Resize frame to HUD resolution if necessary
        if base_frame.shape[1] != self.width or base_frame.shape[0] != self.height:
            frame = cv2.resize(base_frame, (self.width, self.height))
        else:
            frame = base_frame.copy()

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]

        # 1. Overlay sparse optical flow vector quivers
        if flow_u is not None and flow_v is not None:
            fh, fw = flow_u.shape
            scale_x = w / fw
            scale_y = h / fh
            step = 16
            for y in range(0, fh, step):
                for x in range(0, fw, step):
                    dx = flow_u[y, x] * 3.0 * scale_x
                    dy = flow_v[y, x] * 3.0 * scale_y
                    if abs(dx) + abs(dy) > 1.5:
                        pt1 = (int(x * scale_x), int(y * scale_y))
                        pt2 = (int(pt1[0] + dx), int(pt1[1] + dy))
                        cv2.arrowedLine(frame, pt1, pt2, (0, 255, 255), 1, tipLength=0.3)

        # 2. Draw Top Telemetry Bar (Dark semi-transparent header)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Flight mode badge
        mode_colors = {
            FlightMode.CRUISE: (0, 200, 0),
            FlightMode.GAP_SEEKING: (0, 220, 220),
            FlightMode.EVASIVE_SACCADE: (0, 0, 255),
            FlightMode.BRAKING_HOVER: (255, 140, 0),
        }
        color = mode_colors.get(mode, (255, 255, 255))
        cv2.putText(frame, f"MODE: {mode.name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Speed, Alt, Yaw
        telemetry_str = f"SPD: {speed_m_s:4.1f}m/s | ALT: {altitude_m:4.1f}m | HDG: {yaw_deg:4.0f}deg"
        cv2.putText(frame, telemetry_str, (w // 2 - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

        # Safety State badge
        safe_color = (0, 220, 0) if safety_state == EmergencyState.NOMINAL else (0, 50, 255)
        cv2.putText(frame, f"SAFETY: {safety_state.name}", (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, safe_color, 2)

        # 3. Giant Fiber Neural Meter (Bottom Left)
        if gf_state is not None:
            cv2.rectangle(frame, (12, h - 85), (200, h - 12), (30, 30, 35), -1)
            cv2.rectangle(frame, (12, h - 85), (200, h - 12), (80, 80, 90), 1)

            cv2.putText(frame, "GIANT FIBER (GF) LIF", (18, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 220), 1)

            # Potential bar (-70 mV to -40 mV)
            v = gf_state.membrane_potential_mv
            v_norm = float(np.clip((v - (-70.0)) / 25.0, 0.0, 1.0))
            bar_w = int(v_norm * 170)
            bar_color = (0, 0, 255) if gf_state.is_spiking else ((0, 140, 255) if v_norm > 0.75 else (0, 200, 0))

            cv2.rectangle(frame, (18, h - 60), (18 + bar_w, h - 45), bar_color, -1)
            cv2.rectangle(frame, (18, h - 60), (188, h - 45), (120, 120, 120), 1)

            cv2.putText(frame, f"Vm: {v:5.1f} mV", (18, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)

            if gf_state.is_spiking:
                # Big flash indicator
                cv2.putText(frame, "SACCADE SPIKE!", (18, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # 4. Sector Risk / Nearness Gauge (Bottom Center)
        if risk_map is not None:
            num_sec = len(risk_map.sector_risks)
            gauge_w = 200
            gauge_x = (w - gauge_w) // 2
            gauge_y = h - 50
            sec_w = gauge_w // num_sec

            cv2.rectangle(frame, (gauge_x - 4, gauge_y - 20), (gauge_x + gauge_w + 4, h - 12), (30, 30, 35), -1)
            cv2.putText(frame, f"RISK: {risk_map.overall_threat_level}", (gauge_x, gauge_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            for i, r in enumerate(risk_map.sector_risks):
                bx = gauge_x + i * sec_w
                bh = int(r * 30)
                # Color green -> yellow -> red
                red = int(np.clip(r * 2.0 * 255, 0, 255))
                green = int(np.clip((1.0 - r) * 2.0 * 255, 0, 255))
                cv2.rectangle(frame, (bx, gauge_y + 30 - bh), (bx + sec_w - 2, gauge_y + 30), (0, green, red), -1)

        return frame
