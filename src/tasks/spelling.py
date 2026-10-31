"""
Alphabet Spelling Task for FlyBody.

Defines:
- 26 scattered 3D spatial points corresponding to letters 'A' through 'Z' in the arena.
- SpellingTask: Sequences of target letters representing words to spell (e.g., 'UDIT', 'FLY', 'AI').
- Sequential letter contact verification and spelling state management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class LetterTarget:
    """Representation of an alphabet letter target in 3D arena space."""
    letter: str
    position: np.ndarray  # [x, y, z] in cm
    radius: float = 0.15   # 1.5 mm contact radius

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - point))

    def is_contact(self, point: np.ndarray) -> bool:
        return bool(self.distance_to(point) <= self.radius)


class AlphabetArena:
    """
    Scatters 26 spatial points corresponding to the alphabet 'A' to 'Z'
    across the 12 cm x 12 cm x 6 cm FlyBody flight arena.
    """

    def __init__(self, contact_radius: float = 0.15):
        self.contact_radius = contact_radius
        self.letter_targets: Dict[str, LetterTarget] = {}
        self._init_alphabet_points()

    def _init_alphabet_points(self):
        """
        Distributes 26 letter points in the arena forward field of view (X in [0.45, 0.75], Y in [-0.40, +0.40], Z in [0.50, 0.75]).
        Ensures clear spatial distinction while remaining in the reachable flight corridor.
        """
        # Large spaced 3D layout (~2.0 to 3.5 cm separation) across 12cm x 12cm arena
        coords = {
            'A': np.array([6.97, -1.00, 1.35], dtype=np.float32),
            'B': np.array([0.80,  5.00, 1.35], dtype=np.float32),
            'C': np.array([3.64, -5.00, 1.35], dtype=np.float32),
            'D': np.array([0.80,  2.76, 1.35], dtype=np.float32),
            'E': np.array([2.56, -0.42, 1.35], dtype=np.float32),
            'F': np.array([4.69,  5.00, 1.35], dtype=np.float32),
            'G': np.array([2.71,  5.00, 0.75], dtype=np.float32),
            'H': np.array([7.01, -3.18, 1.35], dtype=np.float32),
            'I': np.array([9.20, -5.00, 0.75], dtype=np.float32),
            'J': np.array([0.80,  0.64, 0.75], dtype=np.float32),
            'K': np.array([0.80, -5.00, 1.35], dtype=np.float32),
            'L': np.array([0.80, -2.09, 0.75], dtype=np.float32),
            'M': np.array([9.20,  1.00, 1.35], dtype=np.float32),
            'N': np.array([9.20, -1.05, 0.75], dtype=np.float32),
            'O': np.array([4.63,  1.12, 1.35], dtype=np.float32),
            'P': np.array([9.20, -3.08, 1.35], dtype=np.float32),
            'Q': np.array([4.78, -1.00, 0.75], dtype=np.float32),
            'R': np.array([4.60, -2.86, 1.35], dtype=np.float32),
            'S': np.array([9.20,  5.00, 1.35], dtype=np.float32),
            'T': np.array([9.20,  3.00, 0.75], dtype=np.float32),
            'U': np.array([3.12,  2.59, 1.35], dtype=np.float32),
            'V': np.array([7.03,  1.10, 0.75], dtype=np.float32),
            'W': np.array([2.45, -3.18, 0.75], dtype=np.float32),
            'X': np.array([7.02,  5.00, 0.75], dtype=np.float32),
            'Y': np.array([6.29, -5.00, 0.75], dtype=np.float32),
            'Z': np.array([6.22,  3.04, 1.35], dtype=np.float32),
        }

        # Build LetterTarget objects
        for letter, pos in coords.items():
            self.letter_targets[letter] = LetterTarget(
                letter=letter,
                position=pos,
                radius=self.contact_radius
            )

    def get_target(self, letter: str) -> LetterTarget:
        letter = letter.upper()
        if letter not in self.letter_targets:
            raise KeyError(f"Letter '{letter}' not found in alphabet arena.")
        return self.letter_targets[letter]

    def all_targets(self) -> Dict[str, LetterTarget]:
        return self.letter_targets


@dataclass
class SpellingTask:
    """Manages the mission of spelling a given word letter by letter."""
    word: str
    current_index: int = 0
    spelled_letters: List[str] = field(default_factory=list)
    completed: bool = False

    @property
    def target_letter(self) -> Optional[str]:
        if self.current_index < len(self.word):
            return self.word[self.current_index]
        return None

    def advance_letter(self) -> bool:
        """Records the current letter as spelled and moves to the next."""
        if self.current_index < len(self.word):
            self.spelled_letters.append(self.word[self.current_index])
            self.current_index += 1
            if self.current_index >= len(self.word):
                self.completed = True
        return self.completed

    @property
    def current_spelling_str(self) -> str:
        return "".join(self.spelled_letters)

    @property
    def progress_str(self) -> str:
        res = []
        for i, ch in enumerate(self.word):
            if i < self.current_index:
                res.append(f"[{ch}]")
            elif i == self.current_index:
                res.append(f"*{ch}*")
            else:
                res.append(f"_{ch}_")
        return " ".join(res)
