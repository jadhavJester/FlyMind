"""
Milestone 3: Drosophila Arithmetic Selector v0.1 - Embodied Environment.

Implements Layer 3 (Navigation / Flight) and Layer 4 (FlyBody MuJoCo Physics):
- Integrates the official FlyBody simulation (`flybody.fly_envs.flight_imitation`).
- Maintains a 3D Arena (~12 cm x 12 cm x 6 cm).
- Instantiates two spatial target objects (target_left and target_right) with physical contact detection.
- Bridges the selected target into FlyBody's reference lookahead trajectory interface.
- Calculates physical contact, boundary collisions, timeouts, and rewards strictly from MuJoCo physics state.

Scientific Note:
- Body/proprioceptive observations are physical sensory-like measurements.
- Reference displacement/quaternion vectors are environment-provided task/reference signals,
  NOT visual perception. No claim of visual scene understanding is made.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

import flybody
from flybody import fly_envs
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from src.tasks.arithmetic import ArithmeticTask
from src.data.flight_dataset import quat_to_euler


@dataclass
class SpatialTarget:
    """Representation of a physical target waypoint in the 3D arena."""
    target_id: str
    label: str
    value: int
    position: np.ndarray  # [x, y, z] in cm
    radius: float = 0.15   # 1.5 mm contact sphere radius

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - point))

    def is_contact(self, point: np.ndarray) -> bool:
        return bool(self.distance_to(point) <= self.radius)


class Arena3D:
    """Defines the spatial geometry and boundaries for the 12 cm x 12 cm x 6 cm arena."""

    def __init__(
        self,
        x_bounds: Tuple[float, float] = (-2.0, 10.0),   # 12 cm span
        y_bounds: Tuple[float, float] = (-6.0, 6.0),    # 12 cm span
        z_bounds: Tuple[float, float] = (0.20, 6.0),    # 5.8 cm height span (0.2 cm floor)
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def is_within_bounds(self, point: np.ndarray) -> bool:
        x, y, z = point[0], point[1], point[2]
        return bool(
            self.x_bounds[0] <= x <= self.x_bounds[1] and
            self.y_bounds[0] <= y <= self.y_bounds[1] and
            self.z_bounds[0] <= z <= self.z_bounds[1]
        )


class InteractiveArithmeticEnv:
    """
    Milestone 3 Embodied Arithmetic Flight Environment.
    Wraps FlyBody with spatial targets, deterministic task injection, and physical contact detection.
    """

    def __init__(
        self,
        target_distance_x: float = 0.60,    # 6.0 mm forward
        target_offset_y: float = 0.28,      # +/- 2.8 mm lateral separation (5.6 mm between targets)
        target_height_z: float = 0.60,      # 6.0 mm altitude
        contact_radius: float = 0.15,       # 1.5 mm contact radius
        max_steps: int = 300,               # Max control steps (0.06 s physics @ 5 kHz)
        step_penalty: float = -0.01,
        reward_correct: float = 10.0,
        reward_wrong: float = -10.0,
        reward_collision: float = -20.0,
        reward_timeout: float = -1.0,
    ):
        self.arena = Arena3D()
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_collision = reward_collision
        self.reward_timeout = reward_timeout
        self.contact_radius = contact_radius

        # Target definitions in FlyBody CGS coordinate system
        # Standard FruitFly coords: +X forward, +Y left, -Y right, +Z upward
        self.target_left_pos = np.array([target_distance_x, +target_offset_y, target_height_z], dtype=np.float32)
        self.target_right_pos = np.array([target_distance_x, -target_offset_y, target_height_z], dtype=np.float32)

        # Underlying FlyBody environment using InferenceFlightTrajectoryLoader
        self.underlying_env = fly_envs.flight_imitation(
            ref_path=None,
            terminal_com_dist=float('inf')  # Handled by custom contact/arena logic
        )
        self.dt = self.underlying_env.control_timestep()  # 0.0002 s (5000 Hz)

        self.current_task: Optional[ArithmeticTask] = None
        self.targets: Dict[str, SpatialTarget] = {}
        self.selected_target_id: Optional[str] = None
        self.step_count = 0
        self.fly_trajectory: List[np.ndarray] = []
        self.min_dist_correct = float('inf')
        self.min_dist_wrong = float('inf')

    @property
    def physics(self):
        return self.underlying_env.physics

    def reset(
        self,
        task: ArithmeticTask,
        selected_target_id: str
    ) -> Dict[str, Any]:
        """
        Resets environment with a specific task and selected target waypoint.
        Generates the target-directed lookahead reference trajectory in FlyBody.
        """
        self.current_task = task
        self.selected_target_id = selected_target_id
        self.step_count = 0
        self.fly_trajectory = []
        self.min_dist_correct = float('inf')
        self.min_dist_wrong = float('inf')

        # Instantiate physical targets with values assigned from task
        left_val = task.candidate_target_mapping.get("target_left", task.candidates[0])
        right_val = task.candidate_target_mapping.get("target_right", task.candidates[1])

        self.targets = {
            "target_left": SpatialTarget(
                target_id="target_left",
                label=f"[{left_val}]",
                value=left_val,
                position=self.target_left_pos.copy(),
                radius=self.contact_radius
            ),
            "target_right": SpatialTarget(
                target_id="target_right",
                label=f"[{right_val}]",
                value=right_val,
                position=self.target_right_pos.copy(),
                radius=self.contact_radius
            )
        }

        # Determine target waypoint and heading angle
        target_obj = self.targets[selected_target_id]
        heading_angle = float(np.arctan2(target_obj.position[1], target_obj.position[0]))

        # Generate target-directed reference trajectory for FlyBody controller
        # n_steps = max_steps + 100 to ensure full lookahead coverage
        qpos, qvel = constant_speed_trajectory(
            n_steps=self.max_steps + 200,
            speed=20.0,  # 20 cm/s
            init_pos=(0.0, 0.0, 1.037),
            init_heading=heading_angle,
            body_rot_angle_y=-47.5,
            control_timestep=self.dt
        )
        self.underlying_env.task._traj_generator.set_next_trajectory(qpos, qvel)

        # Reset MuJoCo physics and FlyBody state
        timestep = self.underlying_env.reset()

        initial_fly_pos = self.get_fly_position()
        self.fly_trajectory.append(initial_fly_pos.copy())

        obs = self.get_observation()
        return obs

    def get_fly_position(self) -> np.ndarray:
        """Reads instantaneous center-of-mass/thorax position of FruitFly from MuJoCo."""
        return self.physics.named.data.xpos['walker/thorax'].copy()

    def get_fly_velocity(self) -> np.ndarray:
        """Reads instantaneous linear velocity of FruitFly from MuJoCo."""
        return self.physics.data.qvel[:3].copy()

    def get_observation(self) -> Dict[str, Any]:
        """
        Gathers complete state observation.
        Includes proprioception/kinematics and environment reference cues.
        """
        physics = self.physics
        fly_qvel = physics.data.qvel[:6]
        fly_lin_vel = fly_qvel[:3].copy()
        fly_ang_vel = fly_qvel[3:6].copy()

        fly_qpos = physics.data.qpos[:7]
        fly_pos = fly_qpos[:3].copy()
        fly_quat = fly_qpos[3:7].copy()
        fly_euler = quat_to_euler(fly_quat)

        # Lookahead vectors to reference trajectory
        future_disp = []
        cur_step = self.step_count
        traj_gen = self.underlying_env.task._traj_generator
        com_qpos = getattr(traj_gen, '_com_qpos', None)

        for k in range(1, 6):
            target_step = cur_step + k
            if com_qpos is not None and isinstance(com_qpos, np.ndarray) and target_step < com_qpos.shape[0]:
                ref_point = com_qpos[target_step, :3]
            else:
                ref_point = self.targets[self.selected_target_id].position
            future_disp.extend(ref_point - fly_pos)

        raw_feature = np.concatenate([
            fly_lin_vel,
            fly_ang_vel,
            fly_quat,
            fly_euler,
            np.array(future_disp, dtype=np.float32)
        ]).astype(np.float32)

        return {
            "fly_pos": fly_pos,
            "fly_quat": fly_quat,
            "fly_lin_vel": fly_lin_vel,
            "fly_ang_vel": fly_ang_vel,
            "policy_input_feature": raw_feature,
            "selected_target_id": self.selected_target_id,
            "distances": {
                "target_left": self.targets["target_left"].distance_to(fly_pos),
                "target_right": self.targets["target_right"].distance_to(fly_pos),
            }
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Steps the live FlyBody physics simulation forward by 1 control update (4 MuJoCo substeps).
        Evaluates physical contact with targets, collisions, boundary exit, and reward.
        """
        self.step_count += 1

        # Step FlyBody
        timestep = self.underlying_env.step(action)

        fly_pos = self.get_fly_position()
        self.fly_trajectory.append(fly_pos.copy())

        # Distances to both targets
        d_left = self.targets["target_left"].distance_to(fly_pos)
        d_right = self.targets["target_right"].distance_to(fly_pos)

        correct_id = self.current_task.correct_target_id
        d_correct = d_left if correct_id == "target_left" else d_right
        d_wrong = d_right if correct_id == "target_left" else d_left

        self.min_dist_correct = min(self.min_dist_correct, d_correct)
        self.min_dist_wrong = min(self.min_dist_wrong, d_wrong)

        # Check physical contact
        left_contact = self.targets["target_left"].is_contact(fly_pos)
        right_contact = self.targets["target_right"].is_contact(fly_pos)

        reward = self.step_penalty
        terminated = False
        termination_reason = "STEP"
        success = False

        if left_contact or right_contact:
            contacted_id = "target_left" if left_contact else "target_right"
            terminated = True
            if contacted_id == correct_id:
                reward = self.reward_correct
                termination_reason = "CORRECT_TARGET_CONTACT"
                success = True
            else:
                reward = self.reward_wrong
                termination_reason = "WRONG_TARGET_CONTACT"
                success = False

        elif fly_pos[2] < self.arena.z_bounds[0]:
            # Ground collision / crash
            reward = self.reward_collision
            terminated = True
            termination_reason = "GROUND_COLLISION"
            success = False

        elif not self.arena.is_within_bounds(fly_pos):
            # Arena boundary violation
            reward = self.reward_collision
            terminated = True
            termination_reason = "ARENA_BOUNDARY_VIOLATION"
            success = False

        elif self.step_count >= self.max_steps or timestep.last():
            # Episode timeout
            reward = self.reward_timeout
            terminated = True
            termination_reason = "TIMEOUT"
            success = False

        obs = self.get_observation()
        info = {
            "step": self.step_count,
            "fly_pos": fly_pos.tolist(),
            "dist_left": float(d_left),
            "dist_right": float(d_right),
            "min_dist_correct": float(self.min_dist_correct),
            "min_dist_wrong": float(self.min_dist_wrong),
            "terminated": terminated,
            "termination_reason": termination_reason,
            "success": success,
            "reward": float(reward)
        }

        return obs, float(reward), terminated, info
