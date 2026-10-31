"""
Unit tests for Phase 8 RL Environment & Policy architectures.
"""

import pytest
import numpy as np
import torch
from src.environments.flight_stabilization_env import FlightStabilizationEnv
from training.train_rl import ActorCritic


def test_rl_environment_reset():
    env = FlightStabilizationEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16,)
    assert not np.isnan(obs).any()


def test_rl_environment_step():
    env = FlightStabilizationEnv()
    obs, _ = env.reset(seed=42)
    action = np.zeros(12, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert next_obs.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "pos_err_cm" in info


def test_actor_critic_dimensions():
    model = ActorCritic(obs_dim=16, action_dim=12)
    dummy_input = torch.randn(4, 16)
    action, log_prob, value = model.act(dummy_input[0:1])
    
    assert action.shape == (12,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
