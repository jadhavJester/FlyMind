"""
Unit tests for the Alphabet Spelling Task.
"""

import pytest
import numpy as np

from src.tasks.spelling import AlphabetArena, LetterTarget, SpellingTask


def test_alphabet_arena_initialization():
    arena = AlphabetArena()
    all_targets = arena.all_targets()

    # Must contain all 26 uppercase English letters
    assert len(all_targets) == 26
    for c in range(ord('A'), ord('Z') + 1):
        letter = chr(c)
        assert letter in all_targets
        target = arena.get_target(letter)
        assert isinstance(target.position, np.ndarray)
        assert target.position.shape == (3,)
        # Check coordinates are within the 12cm x 12cm arena boundaries
        assert 0.3 <= target.position[0] <= 10.0
        assert -6.0 <= target.position[1] <= 6.0
        assert 0.3 <= target.position[2] <= 2.0

    # Verify large spatial separation across all 26 letters
    pts = [t.position for t in all_targets.values()]
    min_dist = min(np.linalg.norm(pts[i] - pts[j]) for i in range(26) for j in range(i + 1, 26))
    assert min_dist >= 1.9, f"Minimum distance too small: {min_dist:.2f} cm"


def test_spelling_task_progression_udit():
    task = SpellingTask(word="UDIT")

    assert task.word == "UDIT"
    assert task.target_letter == "U"
    assert task.current_spelling_str == ""
    assert task.completed is False

    # 1. Collect 'U'
    task.advance_letter()
    assert task.current_spelling_str == "U"
    assert task.target_letter == "D"
    assert task.completed is False

    # 2. Collect 'D'
    task.advance_letter()
    assert task.current_spelling_str == "UD"
    assert task.target_letter == "I"
    assert task.completed is False

    # 3. Collect 'I'
    task.advance_letter()
    assert task.current_spelling_str == "UDI"
    assert task.target_letter == "T"
    assert task.completed is False

    # 4. Collect 'T'
    task.advance_letter()
    assert task.current_spelling_str == "UDIT"
    assert task.target_letter is None
    assert task.completed is True


def test_letter_target_contact():
    target = LetterTarget(letter="U", position=np.array([0.60, 0.25, 0.55]), radius=0.15)

    contact_p = np.array([0.62, 0.26, 0.54])
    far_p = np.array([0.60, -0.25, 0.55])

    assert target.is_contact(contact_p) is True
    assert target.is_contact(far_p) is False
