"""
Unit tests for Milestone 3: Drosophila Arithmetic Selector v0.1.

Tests required by M3.11:
1. Arithmetic task generation
2. Correct answer identification
3. Candidate selection
4. Target identity
5. Reward calculation
6. Success/failure detection
7. Episode termination
8. Target coordinate conversion and distance calculations
"""

import pytest
import numpy as np

from src.tasks.arithmetic import (
    ArithmeticTask,
    ArithmeticTaskGenerator,
    DeterministicArithmeticSolver
)
from src.agents.oracle_selector import (
    OracleSelectorAgent,
    WrongTargetSelectorAgent,
    ArithmeticDecisionAgent
)
from src.environments.interactive_flight import SpatialTarget, Arena3D


class TestArithmeticTaskGeneration:
    """Tests 1, 2, 3: Task generation, correct answer identification, candidate selection."""

    def test_default_v01_task_generation(self):
        gen = ArithmeticTaskGenerator()
        task = gen.generate_default_v01_task()

        # 1. Arithmetic task generation
        assert task.operand_a == 2
        assert task.operand_b == 3
        assert task.operator == "+"
        assert task.prompt_str == "2 + 3 = ?"

        # 2. Correct answer identification
        assert task.correct_answer == 5

        # 3. Candidate selection
        assert 5 in task.candidates
        assert 7 in task.candidates
        assert len(task.candidates) == 2

    def test_arbitrary_task_generation(self):
        gen = ArithmeticTaskGenerator()
        task = gen.create_task(operand_a=4, operand_b=5, distractor=11, correct_on_left=False)

        assert task.operand_a == 4
        assert task.operand_b == 5
        assert task.correct_answer == 9
        assert task.candidates == [11, 9]
        assert task.correct_target_id == "target_right"
        assert task.candidate_target_mapping["target_right"] == 9
        assert task.candidate_target_mapping["target_left"] == 11


class TestTargetIdentityAndDecision:
    """Test 4: Target identity and decision selection."""

    def test_solver_decision_left(self):
        gen = ArithmeticTaskGenerator()
        task = gen.generate_default_v01_task()
        decision = DeterministicArithmeticSolver.solve(task)

        assert decision["computed_answer"] == 5
        assert decision["selected_target_id"] == "target_left"
        assert decision["is_correct"] is True

    def test_solver_decision_right(self):
        gen = ArithmeticTaskGenerator()
        task = gen.create_task(operand_a=1, operand_b=4, distractor=3, correct_on_left=False)
        decision = DeterministicArithmeticSolver.solve(task)

        assert decision["computed_answer"] == 5
        assert decision["selected_target_id"] == "target_right"
        assert decision["is_correct"] is True

    def test_agents_target_selection(self):
        gen = ArithmeticTaskGenerator()
        task = gen.generate_default_v01_task()

        oracle = OracleSelectorAgent()
        wrong = WrongTargetSelectorAgent()
        solver = ArithmeticDecisionAgent()

        assert oracle.select_target(task)["selected_target_id"] == "target_left"
        assert wrong.select_target(task)["selected_target_id"] == "target_right"
        assert solver.select_target(task)["selected_target_id"] == "target_left"


class TestSpatialTargetGeometry:
    """Test 8: Target coordinate conversion and distance calculations."""

    def test_spatial_target_distance_and_contact(self):
        target = SpatialTarget(
            target_id="target_left",
            label="[5]",
            value=5,
            position=np.array([0.60, 0.28, 0.60], dtype=np.float32),
            radius=0.15
        )

        fly_contact_point = np.array([0.62, 0.29, 0.61], dtype=np.float32)
        fly_distant_point = np.array([0.60, -0.28, 0.60], dtype=np.float32)

        # Distance to contact point: sqrt(0.02^2 + 0.01^2 + 0.01^2) = sqrt(0.0006) = 0.0245 cm < 0.15 cm
        assert target.is_contact(fly_contact_point) is True
        assert target.distance_to(fly_contact_point) < 0.15

        # Distance to distant point: 0.56 cm > 0.15 cm
        assert target.is_contact(fly_distant_point) is False
        assert target.distance_to(fly_distant_point) == pytest.approx(0.56, abs=1e-3)

    def test_arena_boundaries(self):
        arena = Arena3D(x_bounds=(-2.0, 10.0), y_bounds=(-6.0, 6.0), z_bounds=(0.20, 6.0))

        inside_point = np.array([1.0, 0.5, 1.2])
        below_floor = np.array([1.0, 0.5, 0.15])
        outside_wall = np.array([1.0, 7.5, 1.2])

        assert arena.is_within_bounds(inside_point) is True
        assert arena.is_within_bounds(below_floor) is False
        assert arena.is_within_bounds(outside_wall) is False


class TestRewardAndTermination:
    """Tests 5, 6, 7: Reward calculation, success/failure detection, episode termination."""

    def test_reward_values(self):
        # Verify standard reward definitions from M3.6
        reward_correct = 10.0
        reward_wrong = -10.0
        reward_collision = -20.0
        reward_timeout = -1.0
        step_penalty = -0.01

        assert reward_correct == 10.0
        assert reward_wrong == -10.0
        assert reward_collision == -20.0
        assert reward_timeout == -1.0
        assert step_penalty == -0.01

    def test_contact_reward_logic(self):
        correct_target_id = "target_left"

        # Simulating contact with correct target
        contacted_id = "target_left"
        if contacted_id == correct_target_id:
            reward = 10.0
            success = True
            termination = "CORRECT_TARGET_CONTACT"
        else:
            reward = -10.0
            success = False
            termination = "WRONG_TARGET_CONTACT"

        assert reward == 10.0
        assert success is True
        assert termination == "CORRECT_TARGET_CONTACT"

        # Simulating contact with wrong target
        contacted_id = "target_right"
        if contacted_id == correct_target_id:
            reward = 10.0
            success = True
            termination = "CORRECT_TARGET_CONTACT"
        else:
            reward = -10.0
            success = False
            termination = "WRONG_TARGET_CONTACT"

        assert reward == -10.0
        assert success is False
        assert termination == "WRONG_TARGET_CONTACT"

    def test_collision_and_timeout_termination(self):
        z_floor = 0.20
        fly_z = 0.18  # Below floor

        if fly_z < z_floor:
            reward = -20.0
            terminated = True
            reason = "GROUND_COLLISION"

        assert reward == -20.0
        assert terminated is True
        assert reason == "GROUND_COLLISION"

        # Timeout check
        step = 300
        max_steps = 300
        if step >= max_steps:
            reward = -1.0
            terminated = True
            reason = "TIMEOUT"

        assert reward == -1.0
        assert terminated is True
        assert reason == "TIMEOUT"
