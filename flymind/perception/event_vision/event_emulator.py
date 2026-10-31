"""
Neuromorphic Dynamic Vision Sensor (DVS) Emulator.
Generates asynchronous ON/OFF events from intensity variations, modeling event-based UAV avoidance (EVFly / TU Delft Speck).

Physics model:
An event is triggered at pixel (x, y) whenever the change in logarithmic luminance exceeds a threshold C:
    Delta ln(I(x, y, t)) = ln(I(x, y, t)) - ln(I(x, y, t_last)) >= C_pos  --> ON event (+1)
    Delta ln(I(x, y, t)) - ln(I(x, y, t_last)) <= -C_neg --> OFF event (-1)
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class EventPacket:
    """Represents a set of neuromorphic events accumulated over a time step."""
    xs: np.ndarray      # Array of integer x pixel coordinates (N,)
    ys: np.ndarray      # Array of integer y pixel coordinates (N,)
    polarities: np.ndarray  # Array of +1 (ON) or -1 (OFF) polarities (N,)
    timestamps: np.ndarray  # Array of event timestamps (N,)
    event_count: int
    density_map: np.ndarray  # 2D frame of integrated event counts (H, W)


class DVSEventEmulator:
    """
    Simulates a silicon retina / DVS sensor from high-rate intensity frames.
    """
    def __init__(
        self,
        contrast_threshold: float = 0.20,
        refractory_period_s: float = 1e-4,
        eps: float = 1.0,
    ):
        self.contrast_threshold = contrast_threshold
        self.refractory_period_s = refractory_period_s
        self.eps = eps

        self._last_log_intensity: Optional[np.ndarray] = None
        self._last_event_time: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_log_intensity = None
        self._last_event_time = None

    def generate_events(self, image: np.ndarray, timestamp: float) -> EventPacket:
        """
        Produce events between previous and current image frames.
        Args:
            image: Visual frame (H, W) or (H, W, 3).
            timestamp: Sensor timestamp in seconds.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Logarithmic luminance with epsilon floor
        log_i = np.log(gray.astype(np.float32) + self.eps)
        h, w = gray.shape

        if self._last_log_intensity is None or self._last_log_intensity.shape != log_i.shape:
            self._last_log_intensity = log_i
            self._last_event_time = np.full((h, w), -1.0, dtype=np.float32)
            zeros = np.zeros(0, dtype=np.int32)
            return EventPacket(
                xs=zeros,
                ys=zeros,
                polarities=zeros,
                timestamps=np.zeros(0, dtype=np.float32),
                event_count=0,
                density_map=np.zeros((h, w), dtype=np.float32),
            )

        # Temporal log-difference
        delta_log = log_i - self._last_log_intensity

        # Find positive and negative threshold crossings
        on_mask = delta_log >= self.contrast_threshold
        off_mask = delta_log <= -self.contrast_threshold

        # Enforce refractory period
        if self._last_event_time is not None:
            time_since_last = timestamp - self._last_event_time
            valid_time = time_since_last >= self.refractory_period_s
            on_mask &= valid_time
            off_mask &= valid_time

        on_ys, on_xs = np.where(on_mask)
        off_ys, off_xs = np.where(off_mask)

        xs = np.concatenate([on_xs, off_xs]).astype(np.int32)
        ys = np.concatenate([on_ys, off_ys]).astype(np.int32)
        polarities = np.concatenate([
            np.ones(len(on_xs), dtype=np.int8),
            -np.ones(len(off_xs), dtype=np.int8),
        ])
        timestamps = np.full(len(xs), timestamp, dtype=np.float32)

        # Accumulate 2D density map (H, W)
        density_map = np.zeros((h, w), dtype=np.float32)
        density_map[on_mask] += 1.0
        density_map[off_mask] += 1.0

        # Update reference state
        # Only update pixels where an event actually fired to maintain true cumulative delta
        event_fired_mask = on_mask | off_mask
        self._last_log_intensity[event_fired_mask] = log_i[event_fired_mask]
        self._last_event_time[event_fired_mask] = timestamp

        return EventPacket(
            xs=xs,
            ys=ys,
            polarities=polarities,
            timestamps=timestamps,
            event_count=len(xs),
            density_map=density_map,
        )
