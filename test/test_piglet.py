"""Unit tests for Piglet game mechanics.

Piglet is intentionally small, so these tests exercise the full public rule
surface: spec creation, state construction, stepping, and simple policies.
"""

from __future__ import annotations

import random

import pytest

import optimal_pig.piglet as piglet


class FixedCoin:
    """Small RNG test double for deterministic heads/tails outcomes."""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def test_make_spec_matches_fair_coin_piglet_model():
    # Purpose: verify that Piglet's spec encodes heads as +1 and tails as bust,
    # with probabilities summing to exactly one.
    spec = piglet.piglet_make_spec(target_score=2)

    assert spec["game"] == "piglet"
    assert spec["target_score"] == 2
    assert spec["continue_action"] == "flip"
    assert spec["hold_action"] == "hold"
    assert spec["bust_probability"] == pytest.approx(0.5)
    assert spec["gain_outcomes"] == (1,)
    assert spec["gain_probabilities"] == (0.5,)


def test_make_spec_rejects_too_small_target():
    # Purpose: target scores below 2 do not define a useful Piglet game.
    with pytest.raises(ValueError, match="target_score"):
        piglet.piglet_make_spec(target_score=1)


def test_initial_state_and_legal_actions():
    # Purpose: guard the public constructor and terminal-action behavior.
    state = piglet.piglet_initial_state(first_player=1)
    terminal = piglet.PigletState(scores=(2, 0), turn_total=0, current_player=0, winner=0)

    assert state == piglet.PigletState(scores=(0, 0), turn_total=0, current_player=1, winner=None)
    assert piglet.piglet_legal_actions(state) == ("flip", "hold")
    assert piglet.piglet_legal_actions(terminal) == ()

    with pytest.raises(ValueError, match="first_player"):
        piglet.piglet_initial_state(first_player=-1)


def test_hold_banks_turn_total_and_passes_turn():
    # Purpose: holding should add k to the current player's score and reset k.
    state = piglet.PigletState(scores=(0, 1), turn_total=1, current_player=0)
    nxt = piglet.piglet_step(state, "hold", random.Random(0), target_score=3)

    assert nxt == piglet.PigletState(scores=(1, 1), turn_total=0, current_player=1, winner=None)


def test_hold_that_reaches_target_wins_immediately():
    # Purpose: a winning hold should set winner and leave the winner as current_player.
    state = piglet.PigletState(scores=(1, 0), turn_total=1, current_player=0)
    nxt = piglet.piglet_step(state, "hold", random.Random(0), target_score=2)

    assert nxt == piglet.PigletState(scores=(2, 0), turn_total=0, current_player=0, winner=0)


def test_flip_heads_adds_one_and_keeps_turn():
    # Purpose: heads is represented by random() < 0.5 and increments turn_total.
    state = piglet.PigletState(scores=(0, 0), turn_total=1, current_player=0)
    nxt = piglet.piglet_step(state, "flip", FixedCoin(0.25), target_score=3)

    assert nxt == piglet.PigletState(scores=(0, 0), turn_total=2, current_player=0, winner=None)


def test_flip_tails_busts_and_passes_turn():
    # Purpose: tails should lose the current turn total and pass the turn.
    state = piglet.PigletState(scores=(0, 0), turn_total=1, current_player=0)
    nxt = piglet.piglet_step(state, "flip", FixedCoin(0.75), target_score=3)

    assert nxt == piglet.PigletState(scores=(0, 0), turn_total=0, current_player=1, winner=None)


def test_step_rejects_illegal_action_and_keeps_terminal_state_fixed():
    # Purpose: invalid actions should fail loudly, while terminal states are absorbing.
    state = piglet.piglet_initial_state()
    terminal = piglet.PigletState(scores=(2, 0), turn_total=0, current_player=0, winner=0)

    with pytest.raises(ValueError, match="Illegal Piglet action"):
        piglet.piglet_step(state, "roll", FixedCoin(0.25))  # type: ignore[arg-type]

    assert piglet.piglet_step(terminal, "flip", FixedCoin(0.25), target_score=2) is terminal


def test_threshold_and_always_flip_policies_hold_when_win_is_available():
    # Purpose: simple Piglet policies should never miss an immediate winning hold.
    threshold_policy = piglet.make_hold_at_policy_piglet(threshold=2, target_score=3)
    always_flip_policy = piglet.make_always_flip_policy_piglet(target_score=3)

    assert threshold_policy(piglet.PigletState(scores=(0, 0), turn_total=1, current_player=0), random.Random(0)) == "flip"
    assert threshold_policy(piglet.PigletState(scores=(0, 0), turn_total=2, current_player=0), random.Random(0)) == "hold"
    assert threshold_policy(piglet.PigletState(scores=(2, 0), turn_total=1, current_player=0), random.Random(0)) == "hold"
    assert always_flip_policy(piglet.PigletState(scores=(2, 0), turn_total=1, current_player=0), random.Random(0)) == "hold"
    assert always_flip_policy(piglet.PigletState(scores=(1, 0), turn_total=1, current_player=0), random.Random(0)) == "flip"

    with pytest.raises(ValueError, match="threshold"):
        piglet.make_hold_at_policy_piglet(threshold=-1)
