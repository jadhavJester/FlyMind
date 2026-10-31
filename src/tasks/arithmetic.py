"""
Milestone 3: Drosophila Arithmetic Selector v0.1 - Task & Decision Modules.

Implements:
- Layer 1 (Task): Arithmetic problem generation (e.g., '2 + 3 = ?' with candidates [5, 7]).
- Layer 2 (Decision): Deterministic arithmetic solver matching the correct answer to a target.

Separation of Concerns:
- This module contains zero physics, zero MuJoCo logic, and zero neural networks.
- It operates strictly at the symbolic/arithmetic task representation layer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class ArithmeticTask:
    """Structured representation of an arithmetic problem presented to the fly agent."""
    task_id: str
    operand_a: int
    operand_b: int
    operator: str
    candidates: List[int]
    correct_answer: int
    candidate_target_mapping: Dict[str, int]
    correct_target_id: str
    prompt_str: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "operand_a": self.operand_a,
            "operand_b": self.operand_b,
            "operator": self.operator,
            "candidates": list(self.candidates),
            "correct_answer": self.correct_answer,
            "candidate_target_mapping": dict(self.candidate_target_mapping),
            "correct_target_id": self.correct_target_id,
            "prompt_str": self.prompt_str,
        }


class ArithmeticTaskGenerator:
    """Generates arithmetic tasks with candidate answers and spatial target mappings."""

    def __init__(self, default_target_ids: tuple = ("target_left", "target_right")):
        self.target_left_id = default_target_ids[0]
        self.target_right_id = default_target_ids[1]

    def generate_default_v01_task(self) -> ArithmeticTask:
        """
        Generates the standard Milestone 3 task:
            2 + 3 = ?
        Candidate answers:
            Left:  [5] (Correct)
            Right: [7] (Distractor)
        """
        return self.create_task(
            operand_a=2,
            operand_b=3,
            distractor=7,
            correct_on_left=True,
            task_id="m3_arithmetic_2_plus_3_v01"
        )

    def create_task(
        self,
        operand_a: int,
        operand_b: int,
        distractor: Optional[int] = None,
        correct_on_left: bool = True,
        task_id: Optional[str] = None
    ) -> ArithmeticTask:
        """
        Creates an addition task: operand_a + operand_b = ?
        
        Args:
            operand_a: First integer operand.
            operand_b: Second integer operand.
            distractor: Incorrect answer candidate. If None, generated automatically.
            correct_on_left: If True, correct target is placed on target_left.
            task_id: Optional unique task identifier string.
        """
        correct_answer = operand_a + operand_b
        if distractor is None:
            # Generate plausible distractor (offset by +2 or -1, ensuring positive distinct value)
            offset = 2 if correct_answer % 2 == 0 else -1
            distractor = correct_answer + offset
            if distractor == correct_answer or distractor < 0:
                distractor = correct_answer + 3

        tid = task_id or f"task_{uuid.uuid4().hex[:8]}"
        prompt_str = f"{operand_a} + {operand_b} = ?"

        if correct_on_left:
            mapping = {
                self.target_left_id: correct_answer,
                self.target_right_id: distractor
            }
            candidates = [correct_answer, distractor]
            correct_target_id = self.target_left_id
        else:
            mapping = {
                self.target_left_id: distractor,
                self.target_right_id: correct_answer
            }
            candidates = [distractor, correct_answer]
            correct_target_id = self.target_right_id

        return ArithmeticTask(
            task_id=tid,
            operand_a=operand_a,
            operand_b=operand_b,
            operator="+",
            candidates=candidates,
            correct_answer=correct_answer,
            candidate_target_mapping=mapping,
            correct_target_id=correct_target_id,
            prompt_str=prompt_str
        )


class DeterministicArithmeticSolver:
    """
    Layer 2: Deterministic Task Solver.
    
    Solves A + B = C and matches C against candidate spatial targets.
    Note: For Milestone 3 v0.1, this is an exact symbolic solver rather than a trained
    neural arithmetic network, isolating the embodied task pipeline from learning noise.
    """

    @staticmethod
    def solve(task: ArithmeticTask) -> Dict[str, Any]:
        """
        Evaluates task and selects the target ID matching the correct answer.
        
        Returns:
            Dict containing selected_target_id, computed_answer, and metadata.
        """
        if task.operator == "+":
            computed = task.operand_a + task.operand_b
        elif task.operator == "-":
            computed = task.operand_a - task.operand_b
        elif task.operator == "*":
            computed = task.operand_a * task.operand_b
        else:
            raise ValueError(f"Unsupported operator: {task.operator}")

        selected_target_id = None
        for target_id, val in task.candidate_target_mapping.items():
            if val == computed:
                selected_target_id = target_id
                break

        if selected_target_id is None:
            raise RuntimeError(
                f"Computed answer {computed} not found among candidates: {task.candidate_target_mapping}"
            )

        return {
            "computed_answer": computed,
            "selected_target_id": selected_target_id,
            "is_correct": (selected_target_id == task.correct_target_id),
            "confidence": 1.0,
            "solver_type": "DeterministicArithmeticSolver_v0.1"
        }
