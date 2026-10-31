"""
Vectorized 2D Hassenstein-Reichardt Elementary Motion Detector (HR-EMD).
Models the classic correlation-type motion detector found in the Drosophila optic lobe (Lamina -> Medulla -> Lobula).

Biological pathway:
- Photoreceptors (R1-R6) capture luminance.
- Lamina Monopolar Cells (L1 ON, L2 OFF) perform spatial bandpass filtering.
- Medullary neurons (T4/T5) implement direction-selective correlation with temporal delay tau.
"""
from typing import Dict, Tuple
import cv2
import numpy as np


class HRElementaryMotionDetector:
    """
    2D Hassenstein-Reichardt correlator array.
    Outputs directional motion signals across 4 cardinal directions (Right, Left, Up, Down).
    """
    def __init__(
        self,
        tau: float = 0.05,
        dt: float = 0.02,
        spatial_sigma: float = 1.5,
        spatial_step: int = 2,
    ):
        """
        Args:
            tau: Time constant of the low-pass delay filter in seconds (~30-50 ms in Drosophila).
            dt: Discrete simulation timestep in seconds.
            spatial_sigma: Gaussian blur sigma representing ommatidial acceptance angle (~5.1 deg in fly).
            spatial_step: Pixel separation between adjacent sampling ommatidia (Delta s).
        """
        self.tau = tau
        self.dt = dt
        self.spatial_sigma = spatial_sigma
        self.spatial_step = spatial_step

        # Low-pass filter coefficient: y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
        self.alpha = float(np.clip(dt / (tau + dt), 0.01, 1.0))

        self._delayed_state: np.ndarray | None = None
        self._prev_frame: np.ndarray | None = None

    def reset(self) -> None:
        """Reset temporal delay state buffers."""
        self._delayed_state = None
        self._prev_frame = None

    def process_frame(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process visual frame through the HR-EMD array.
        Args:
            image: 2D grayscale or 3D color visual frame.
        Returns:
            Dictionary containing:
                'right': Positive motion rightward (H, W)
                'left':  Positive motion leftward (H, W)
                'up':    Positive motion upward (H, W)
                'down':  Positive motion downward (H, W)
                'flow_u': Net horizontal motion field (Right - Left)
                'flow_v': Net vertical motion field (Down - Up)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 1. Ommatidial spatial low-pass filter (optics point spread function)
        blurred = cv2.GaussianBlur(
            gray.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=self.spatial_sigma,
            sigmaY=self.spatial_sigma,
        )

        # 2. Lamina high-pass filter (subtract local mean luminance / DC component)
        local_mean = cv2.GaussianBlur(blurred, (0, 0), sigmaX=self.spatial_sigma * 3.0)
        signal = blurred - local_mean

        # Initialize delayed state if first call
        if self._delayed_state is None or self._delayed_state.shape != signal.shape:
            self._delayed_state = signal.copy()
            self._prev_frame = signal.copy()
            zeros = np.zeros_like(signal)
            return {
                "right": zeros,
                "left": zeros,
                "up": zeros,
                "down": zeros,
                "flow_u": zeros,
                "flow_v": zeros,
            }

        # 3. Update temporal low-pass delayed state: tau * dL/dt = -L + I
        self._delayed_state = (1.0 - self.alpha) * self._delayed_state + self.alpha * signal

        s = self.spatial_step
        # 4. Horizontal Reichardt correlator
        # Rightwards: signal[:, :-s] * delayed[:, s:] - signal[:, s:] * delayed[:, :-s]
        r_right = np.zeros_like(signal)
        corr_r = signal[:, :-s] * self._delayed_state[:, s:] - signal[:, s:] * self._delayed_state[:, :-s]
        r_right[:, :-s] = np.maximum(0.0, corr_r)
        r_left = np.zeros_like(signal)
        r_left[:, :-s] = np.maximum(0.0, -corr_r)

        # 5. Vertical Reichardt correlator
        # Downwards: signal[:-s, :] * delayed[s:, :] - signal[s:, :] * delayed[:-s, :]
        r_down = np.zeros_like(signal)
        corr_d = signal[:-s, :] * self._delayed_state[s:, :] - signal[s:, :] * self._delayed_state[:-s, :]
        r_down[:-s, :] = np.maximum(0.0, corr_d)
        r_up = np.zeros_like(signal)
        r_up[:-s, :] = np.maximum(0.0, -corr_d)

        flow_u = r_right - r_left
        flow_v = r_down - r_up

        self._prev_frame = signal.copy()

        return {
            "right": r_right,
            "left": r_left,
            "up": r_up,
            "down": r_down,
            "flow_u": flow_u,
            "flow_v": flow_v,
        }
