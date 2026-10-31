"""
Unit tests for Phase 9 Vision-Guided Environment & Neural Models.
"""

import pytest
import numpy as np
import torch
from src.environments.vision_target_env import VisionTargetEnv
from src.models.vision_policy import CNNVisualEncoder, DirectVisionPolicy, HierarchicalVisionPolicy
from src.models.mlp_policy import MLPPolicy


def test_vision_environment_reset():
    env = VisionTargetEnv()
    obs, info = env.reset(seed=42)
    
    assert "image" in obs
    assert "kinematic" in obs
    assert obs["image"].shape == (64, 64, 3)
    assert obs["kinematic"].shape == (16,)


def test_vision_environment_step():
    env = VisionTargetEnv()
    obs, _ = env.reset(seed=42)
    action = np.zeros(12, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs["image"].shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert "dist_to_target_cm" in info


def test_cnn_encoder_forward():
    encoder = CNNVisualEncoder(latent_dim=32)
    dummy_img = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
    out = encoder(dummy_img)
    assert out.shape == (2, 32)


def test_hierarchical_policy_forward():
    low_level = MLPPolicy(input_dim=28, action_dim=12)
    policy = HierarchicalVisionPolicy(low_level_policy=low_level)

    dummy_img = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
    dummy_kin = torch.randn(2, 16)
    actions = policy(dummy_img, dummy_kin)

    assert actions.shape == (2, 12)
