"""
Vision Neural Network Architectures for BioFlight (Phase 9).
Implements CNN visual encoder, Direct Vision Policy (Architecture A),
and Hierarchical Vision Policy (Architecture B).
"""

import torch
import torch.nn as nn
import numpy as np
from src.models.mlp_policy import MLPPolicy


class CNNVisualEncoder(nn.Module):
    """
    3-layer Convolutional Neural Network encoding 64x64x3 RGB image frames into a 32D latent vector.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),            # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, H, W, C) -> permute to (B, C, H, W) and normalize to [0, 1]
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.conv(x)


class DirectVisionPolicy(nn.Module):
    """
    Architecture A: End-to-End Direct Control
    Visual Image -> CNN Encoder -> 12D Wing Motor Actions directly.
    """

    def __init__(self, action_dim: int = 12, action_scale: float = 3.0):
        super().__init__()
        self.encoder = CNNVisualEncoder(latent_dim=32)
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.action_scale = action_scale

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(img_tensor)
        raw_action = self.head(latent)
        return raw_action * self.action_scale


class HierarchicalVisionPolicy(nn.Module):
    """
    Architecture B: Hierarchical Control
    High-Level: Visual Image -> CNN Encoder -> Desired 3D Target Command (cm)
    Low-Level: Desired 3D Target Command + Kinematics -> Pretrained MLP Policy -> 12D Wing Actions.
    """

    def __init__(self, low_level_policy: MLPPolicy, action_scale: float = 3.0):
        super().__init__()
        self.encoder = CNNVisualEncoder(latent_dim=32)
        # High-level controller predicts 3D target displacement (dx, dy, dz)
        self.high_level_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )
        self.low_level_policy = low_level_policy
        self.action_scale = action_scale

    def forward(self, img_tensor: torch.Tensor, kinematic_tensor: torch.Tensor) -> torch.Tensor:
        # High-level visual guidance prediction
        latent = self.encoder(img_tensor)
        target_disp_pred = self.high_level_head(latent) * 3.0  # Range [-3cm, 3cm]

        # Construct 28D input expected by pretrained low-level MLP policy:
        # [fly_lin_vel(3), fly_ang_vel(3), fly_quat(4), fly_euler(3), 5-step lookahead disp(15)]
        # Replace future lookahead with predicted displacement
        batch_size = img_tensor.shape[0]
        lookahead_15d = target_disp_pred.repeat(1, 5)  # 3D repeated 5 times = 15D

        base_kinematic = kinematic_tensor[:, :13]  # [lin_vel(3), ang_vel(3), quat(4), euler(3)]
        full_28d_obs = torch.cat([base_kinematic, lookahead_15d], dim=-1)

        # Pass through low-level pretrained flight policy
        return self.low_level_policy(full_28d_obs)
