"""
Machine Learning Baseline: Lightweight Monocular Depth / Collision Network.
Represents modern learned collision avoidance architectures for comparison against bio-inspired circuits.
"""
from typing import Tuple
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..navigation.safe_navigation.reactive_navigator import FlightSetpoint
from ..navigation.safe_navigation.reflex_state_machine import FlightMode


if TORCH_AVAILABLE:
    class TinyDepthNet(nn.Module):
        """Minimal 3-layer CNN estimating 8-sector depth clearance directly from grayscale images."""
        def __init__(self, num_sectors: int = 8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # 60x80
                nn.ReLU(),
                nn.MaxPool2d(2),                                        # 30x40
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 15x20
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, num_sectors),
                nn.Sigmoid(),  # Output in [0, 1] relative clearance per sector
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class MonocularDepthBaseline:
    """
    Learned visual navigation baseline.
    Predicts sectoral clearances using a compact CNN and commands steering setpoints.
    """
    def __init__(self, num_sectors: int = 8, nominal_speed: float = 1.5):
        self.num_sectors = num_sectors
        self.nominal_speed = nominal_speed
        self.torch_available = TORCH_AVAILABLE

        if self.torch_available:
            self.model = TinyDepthNet(num_sectors=num_sectors)
            self.model.eval()
        else:
            self.model = None

    def reset(self) -> None:
        pass

    def compute_setpoint(self, image: np.ndarray, current_altitude: float) -> FlightSetpoint:
        """Compute flight setpoint from CNN depth estimation."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        if self.torch_available and self.model is not None:
            # Resize to network input (120, 160)
            resized = cv2.resize(gray, (160, 120))
            tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                clearances = self.model(tensor).squeeze(0).numpy()
        else:
            # Heuristic depth approximation if torch is absent
            # Inverted normalized luminance
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            sector_w = gray.shape[1] // self.num_sectors
            clearances = np.zeros(self.num_sectors, dtype=np.float32)
            for i in range(self.num_sectors):
                xs = i * sector_w
                xe = (i + 1) * sector_w if i < self.num_sectors - 1 else gray.shape[1]
                clearances[i] = float(np.mean(blur[:, xs:xe])) / 255.0

        # Best sector has maximum clearance
        best_idx = int(np.argmax(clearances))
        angles = np.linspace(-np.radians(45), np.radians(45), self.num_sectors)
        target_yaw_rate = float(np.clip(angles[best_idx] * 1.5, -2.0, 2.0))

        # Speed throttle
        min_clearance = float(np.min(clearances))
        vx = max(0.3, self.nominal_speed * min_clearance)

        alt_err = 1.5 - current_altitude
        vz = float(np.clip(-1.0 * alt_err, -0.8, 0.8))

        return FlightSetpoint(
            vx_body=vx,
            vy_body=0.0,
            vz_body=vz,
            yaw_rate=target_yaw_rate,
            mode=FlightMode.CRUISE,
        )
