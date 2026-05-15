"""Unit tests for the one-die Pig game mechanics.

These tests focus on deterministic game rules rather than value-iteration
output. They use explicit submodule imports to avoid the package-level
``import optimal_pig as pig`` name-collision risk.
"""

from __future__ import annotations

import random

import pytest

import optimal_pig.pig as pig


class FixedDie:
    """Small RNG test double that returns a chosen die roll."""

    def __init__(self, roll: int):
        self.roll = roll

    def randint(self, low: int, high: int) -> int:
        assert (low, high) == (1, 6)
        return self.roll


def test_make_spec_matches_one_die_pig_model():
    # Purpose: verify that Pig's value-iteration spec encodes one fair die:
    # bust on 1, gains 2--6, and total transition probability 1.
    spec = pig.make_spec(target_score=100)

    assert spec["game"] == "pig"
    assert spec["target_score"] == 100
    assert spec["continue_action"] == "roll"
    assert spec["hold_action"] == "hold"
    assert spec["bust_probability"] == pytest.approx(1.0 / 6.0)
    assert spec["gain_outcomes"] == (2, 3, 4, 5, 6)
    assert sum(spec["gain_probabilities"]) + spec["bust_probability"] == pytest.approx(1.0)


def test_make_spec_rejects_too_small_target():
    # Purpose: target scores below 2 do not define a meaningful two-player game.
    with pytest.raises(ValueError, match="target_score"):
        pig.make_spec(target_score=1)


def test_initial_state_and_legal_actions():
    # Purpose: guard the public state constructor and terminal-action behavior.
    state = pig.initial_state(first_player=1)
    terminal = pig.PigState(scores=(100, 0), turn_total=0, current_player=0, winner=0)

    assert state == pig.PigState(scores=(0, 0), turn_total=0, current_player=1, winner=None)
    assert pig.legal_actions(state) == ("roll", "hold")
    assert pig.legal_actions(terminal) == ()

    with pytest.raises(ValueError, match="first_player"):
        pig.initial_state(first_player=2)


def test_hold_banks_turn_total_and_passes_turn():
    # Purpose: holding should add only the current turn total, reset k, and pass.
    state = pig.PigState(scores=(10, 20), turn_total=7, current_player=0)
    nxt = pig.step(state, "hold", random.Random(0), target_score=100)

    assert nxt == pig.PigState(scores=(17, 20), turn_total=0, current_player=1, winner=None)


def test_hold_that_reaches_target_wins_immediately():
    # Purpose: a winning hold should set winner and keep the winner as current_player.
    state = pig.PigState(scores=(94, 50), turn_total=6, current_player=0)
    nxt = pig.step(state, "hold", random.Random(0), target_score=100)

    assert nxt.scores == (100, 50)
    assert nxt.turn_total == 0
    assert nxt.current_player == 0
    assert nxt.winner == 0


def test_roll_one_busts_and_passes_turn():
    # Purpose: rolling 1 loses the turn total without changing banked scores.
    state = pig.PigState(scores=(10, 20), turn_total=12, current_player=0)
    nxt = pig.step(state, "roll", FixedDie(1), target_score=100)

    assert nxt == pig.PigState(scores=(10, 20), turn_total=0, current_player=1, winner=None)


def test_roll_two_to_six_adds_to_turn_total_and_keeps_turn():
    # Purpose: non-bust die rolls should accumulate in k and leave the same player to act.
    state = pig.PigState(scores=(10, 20), turn_total=12, current_player=0)
    nxt = pig.step(state, "roll", FixedDie(5), target_score=100)

    assert nxt == pig.PigState(scores=(10, 20), turn_total=17, current_player=0, winner=None)


def test_step_rejects_illegal_action_and_keeps_terminal_state_fixed():
    # Purpose: invalid actions should fail loudly, while terminal states are absorbing.
    state = pig.initial_state()
    terminal = pig.PigState(scores=(100, 0), turn_total=0, current_player=0, winner=0)

    with pytest.raises(ValueError, match="Illegal Pig action"):
        pig.step(state, "flip", random.Random(0))  # type: ignore[arg-type]

    assert pig.step(terminal, "roll", FixedDie(6), target_score=100) is terminal


def test_hold_at_policy_threshold_and_immediate_win():
    # Purpose: the baseline threshold policy should hold at threshold and always
    # hold when banking the current total wins.
    policy = pig.make_hold_at_policy(threshold=20, target_score=100)

    assert policy(pig.PigState(scores=(0, 0), turn_total=19, current_player=0), random.Random(0)) == "roll"
    assert policy(pig.PigState(scores=(0, 0), turn_total=20, current_player=0), random.Random(0)) == "hold"
    assert policy(pig.PigState(scores=(95, 0), turn_total=5, current_player=0), random.Random(0)) == "hold"

    with pytest.raises(ValueError, match="threshold"):
        pig.make_hold_at_policy(threshold=-1)
