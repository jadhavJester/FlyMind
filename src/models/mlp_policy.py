"""
Neural policy architectures for BioFlight (Phase 6).
Implements a clean MLP flight policy with LayerNorm, SiLU activations,
and scaled tanh outputs matching the FlyBody action space.
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    Multi-Layer Perceptron policy mapping flight kinematic observables
    to 12-dimensional FlyBody motor/wing actions.
    """

    def __init__(
        self,
        input_dim: int = 28,
        action_dim: int = 12,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = "silu",
        action_scale: float = 3.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        act_cls = nn.SiLU if activation.lower() == "silu" else nn.ReLU

        layers = []
        layers.append(nn.LayerNorm(input_dim))

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, action_dim)

        # Initialize head weights with small values for smooth initial actions
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        raw_out = self.head(features)
        # Bounded actions scaled to [-action_scale, action_scale]
        return torch.tanh(raw_out) * self.action_scale

    @torch.no_grad()
    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """Runs single-step inference returning a numpy action vector."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
                action = self.forward(state_t).squeeze(0).cpu().numpy()
            else:
                state_t = torch.from_numpy(state).float().to(device)
                action = self.forward(state_t).cpu().numpy()
        else:
            action = self.forward(state.to(device)).cpu().numpy()

        return action
