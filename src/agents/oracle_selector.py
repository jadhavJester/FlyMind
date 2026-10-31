"""
Milestone 3: Drosophila Arithmetic Selector v0.1 - Agent Implementations.

Implements:
1. Target Selection Agents:
   - OracleSelectorAgent: Directly queries ground truth to select correct target (Baseline A).
   - WrongTargetSelectorAgent: Explicitly commands wrong target to test error detection (Baseline B).
   - ArithmeticDecisionAgent: Evaluates problem via DeterministicArithmeticSolver and selects target (Baseline C).

2. FlightPolicyController:
   - Wraps the trained FlyBody neural tracking policy (`checkpoints/imitation/exp_001/best_model.pt`).
   - Computes normalized state vectors and outputs the 12-channel FlyBody action vector.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
try:
    import torch
    from src.models.mlp_policy import MLPPolicy
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from src.tasks.arithmetic import ArithmeticTask, DeterministicArithmeticSolver


class BaseSelectorAgent(ABC):
    """Abstract base class for high-level decision agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_target(self, task: ArithmeticTask) -> Dict[str, Any]:
        """
        Receives an arithmetic task and outputs the selected target ID.
        
        Returns:
            Dict containing selected_target_id, decision_metadata, and agent name.
        """
        pass


class OracleSelectorAgent(BaseSelectorAgent):
    """
    Baseline A: Oracle Agent.
    Bypasses arithmetic solving and directly commands the ground truth correct target.
    Verifies physical flight reachability to the correct target.
    """

    def __init__(self):
        super().__init__(name="OracleSelector_BaselineA")

    def select_target(self, task: ArithmeticTask) -> Dict[str, Any]:
        return {
            "selected_target_id": task.correct_target_id,
            "agent_type": self.name,
            "rationale": "Oracle selection from task ground truth",
            "computed_answer": task.correct_answer
        }


class WrongTargetSelectorAgent(BaseSelectorAgent):
    """
    Baseline B: Adversarial / Incorrect Agent.
    Explicitly selects the incorrect distractor target.
    Verifies target spatial separation, failure detection, and negative reward (-10).
    """

    def __init__(self):
        super().__init__(name="WrongTargetSelector_BaselineB")

    def select_target(self, task: ArithmeticTask) -> Dict[str, Any]:
        wrong_id = None
        for target_id in task.candidate_target_mapping.keys():
            if target_id != task.correct_target_id:
                wrong_id = target_id
                break
        if wrong_id is None:
            wrong_id = task.correct_target_id  # Fallback

        return {
            "selected_target_id": wrong_id,
            "agent_type": self.name,
            "rationale": "Intentional selection of wrong distractor target",
            "computed_answer": task.candidate_target_mapping[wrong_id]
        }


class ArithmeticDecisionAgent(BaseSelectorAgent):
    """
    Baseline C: Complete Embodied Task Agent.
    Invokes Layer 2 DeterministicArithmeticSolver to solve the problem (A + B = C),
    matches against candidate spatial targets, and commands the selected waypoint.
    """

    def __init__(self):
        super().__init__(name="ArithmeticDecisionAgent_BaselineC")

    def select_target(self, task: ArithmeticTask) -> Dict[str, Any]:
        solve_result = DeterministicArithmeticSolver.solve(task)
        return {
            "selected_target_id": solve_result["selected_target_id"],
            "agent_type": self.name,
            "rationale": f"Solved arithmetic: {task.prompt_str} -> {solve_result['computed_answer']}",
            "computed_answer": solve_result["computed_answer"],
            "confidence": solve_result["confidence"]
        }


class NumpyMLPPolicy:
    """Zero-dependency pure-NumPy inference for MLPPolicy."""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.w_ln = data['net.0.weight']
        self.b_ln = data['net.0.bias']
        self.w1 = data['net.1.weight']
        self.b1 = data['net.1.bias']
        self.w2 = data['net.3.weight']
        self.b2 = data['net.3.bias']
        self.w3 = data['net.5.weight']
        self.b3 = data['net.5.bias']
        self.wh = data['head.weight']
        self.bh = data['head.bias']

    def select_action(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5) * self.w_ln + self.b_ln
        
        h1 = x_norm @ self.w1.T + self.b1
        h1 = h1 / (1.0 + np.exp(-np.clip(h1, -20.0, 20.0)))
        
        h2 = h1 @ self.w2.T + self.b2
        h2 = h2 / (1.0 + np.exp(-np.clip(h2, -20.0, 20.0)))
        
        h3 = h2 @ self.w3.T + self.b3
        h3 = h3 / (1.0 + np.exp(-np.clip(h3, -20.0, 20.0)))
        
        out = np.tanh(h3 @ self.wh.T + self.bh) * 3.0
        return out[0]


class FlightPolicyController:
    """
    Layer 3 & 4 Interface:
    Loads trained FlyBody neural flight policy and maps observations to 12-channel flight actions.
    Supports both PyTorch and lightweight pure-NumPy inference.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/imitation/exp_001/best_model.pt",
        metadata_path: str = "data/processed/metadata.json"
    ):
        npz_path = checkpoint_path.replace(".pt", ".npz").replace("best_model", "model_weights")
        if not os.path.exists(checkpoint_path) and not os.path.exists(npz_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {npz_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        # Load normalization parameters
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        mean_val = meta.get("mean") or meta.get("mean_X")
        std_val = meta.get("std") or meta.get("std_X")
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)

        # Prefer high-speed pure-NumPy policy if available
        if os.path.exists(npz_path):
            self.policy = NumpyMLPPolicy(npz_path)
        else:
            self.policy = MLPPolicy(input_dim=28, action_dim=12)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.policy.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.policy.load_state_dict(checkpoint)
            self.policy.eval()

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Receives environment observation dict, extracts the 28-dim state feature,
        normalizes it, and queries the policy network.
        
        Returns:
            np.ndarray: 12-dimensional FlyBody continuous action.
        """
        raw_feature = obs["policy_input_feature"]
        norm_feature = (raw_feature - self.mean) / self.std
        action = self.policy.select_action(norm_feature)
        return action
