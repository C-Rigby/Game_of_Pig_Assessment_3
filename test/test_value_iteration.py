"""Unit tests for the value-iteration solver.

The tests cover both low-level Bellman helpers and full convergence on the
paper's Piglet goal=2 example, where exact probabilities are known.
"""

from __future__ import annotations

import numpy as np
import pytest

import optimal_pig.pig as pig
import optimal_pig.piglet as piglet
import optimal_pig.value_iteration_fun as vi


def test_validate_spec_accepts_pig_and_piglet_specs():
    # Purpose: both game modules should emit specs accepted by the solver.
    vi.validate_spec(pig.make_spec(target_score=10))
    vi.validate_spec(piglet.piglet_make_spec(target_score=2))


@pytest.mark.parametrize(
    "bad_spec, message",
    [
        ({}, "missing required keys"),
        (
            {
                "target_score": 1,
                "continue_action": "flip",
                "hold_action": "hold",
                "bust_probability": 0.5,
                "gain_outcomes": (1,),
                "gain_probabilities": (0.5,),
            },
            "target_score",
        ),
        (
            {
                "target_score": 2,
                "continue_action": "flip",
                "hold_action": "hold",
                "bust_probability": 0.5,
                "gain_outcomes": (1, 2),
                "gain_probabilities": (0.5,),
            },
            "same length",
        ),
        (
            {
                "target_score": 2,
                "continue_action": "flip",
                "hold_action": "hold",
                "bust_probability": 0.5,
                "gain_outcomes": (0,),
                "gain_probabilities": (0.5,),
            },
            "positive integers",
        ),
        (
            {
                "target_score": 2,
                "continue_action": "flip",
                "hold_action": "hold",
                "bust_probability": 0.4,
                "gain_outcomes": (1,),
                "gain_probabilities": (0.4,),
            },
            "sum to 1",
        ),
    ],
)
def test_validate_spec_rejects_malformed_specs(bad_spec, message):
    # Purpose: malformed transition models should fail before iteration starts.
    with pytest.raises(ValueError, match=message):
        vi.validate_spec(bad_spec)


def test_state_iteration_and_count_for_restricted_piglet_goal_two():
    # Purpose: for G=2 the restricted non-terminal state space has exactly six
    # states, matching the worked example in the paper.
    spec = piglet.piglet_make_spec(target_score=2)
    states = list(vi.iter_states(spec, restricted_k=True))

    assert vi.count_states(spec, restricted_k=True) == 6
    assert states == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
    ]


def test_unrestricted_state_iteration_uses_full_cube():
    # Purpose: unrestricted mode is used by the plotting notebooks, so it
    # should include all G^3 padded states.
    spec = piglet.piglet_make_spec(target_score=2)

    assert vi.count_states(spec, restricted_k=False) == 8
    assert len(list(vi.iter_states(spec, restricted_k=False))) == 8


def test_make_value_table_marks_invalid_entries_only_in_restricted_mode():
    # Purpose: restricted tables should keep invalid terminal-padding cells as
    # NaN, while unrestricted tables expose the full cube for plotting.
    spec = piglet.piglet_make_spec(target_score=2)

    restricted = vi.make_value_table(spec, init_value=0.25, restricted_k=True)
    unrestricted = vi.make_value_table(spec, init_value=0.25, restricted_k=False)

    assert restricted.shape == (2, 2, 2)
    assert restricted[0, 0, 0] == pytest.approx(0.25)
    assert np.isnan(restricted[1, 0, 1])
    assert np.all(unrestricted == pytest.approx(0.25))


def test_value_at_returns_one_for_terminal_win_and_rejects_invalid_state():
    # Purpose: Bellman equations depend on value_at treating i+k>=G as an
    # immediate win, without indexing into NaN padding.
    spec = piglet.piglet_make_spec(target_score=2)
    V = vi.make_value_table(spec, init_value=0.0, restricted_k=True)

    assert vi.value_at(spec, V, 1, 0, 1, restricted_k=True) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Invalid non-terminal state"):
        vi.value_at(spec, V, -1, 0, 0, restricted_k=True)


def test_q_values_match_piglet_bellman_formula_on_zero_table():
    # Purpose: check the two Bellman action-value formulas directly before
    # testing the iterative solver.
    spec = piglet.piglet_make_spec(target_score=2)
    V = vi.make_value_table(spec, init_value=0.0, restricted_k=True)

    assert vi.q_continue(spec, V, 0, 0, 0, restricted_k=True) == pytest.approx(0.5)
    assert vi.q_hold(spec, V, 0, 0, 0, restricted_k=True) == pytest.approx(1.0)
    assert vi.bellman_update(spec, V, 0, 0, 0, restricted_k=True) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Invalid state"):
        vi.q_continue(spec, V, 1, 0, 1, restricted_k=True)


def test_best_action_uses_tie_action_when_action_values_equal():
    # Purpose: policy extraction depends on deterministic tie handling.
    spec = piglet.piglet_make_spec(target_score=2)
    V = vi.make_value_table(spec, init_value=0.0, restricted_k=True)

    assert vi.q_continue(spec, V, 1, 0, 0, restricted_k=True) == pytest.approx(
        vi.q_hold(spec, V, 1, 0, 0, restricted_k=True)
    )
    assert vi.best_action(spec, V, 1, 0, 0, tie_action="flip", restricted_k=True) == "flip"
    assert vi.best_action(spec, V, 1, 0, 0, tie_action="hold", restricted_k=True) == "hold"


def test_full_value_iteration_solves_exact_piglet_goal_two_values():
    # Purpose: the paper gives exact Piglet goal=2 probabilities, making this
    # a compact end-to-end regression test for convergence and Bellman updates.
    spec = piglet.piglet_make_spec(target_score=2)
    exact = {
        (0, 0, 0): 4.0 / 7.0,
        (0, 0, 1): 5.0 / 7.0,
        (0, 1, 0): 2.0 / 5.0,
        (0, 1, 1): 3.0 / 5.0,
        (1, 0, 0): 4.0 / 5.0,
        (1, 1, 0): 2.0 / 3.0,
    }

    result = vi.value_iteration(
        spec,
        tol=1e-12,
        max_iterations=10_000,
        trace_states=exact.keys(),
        restricted_k=True,
    )

    assert result["converged"] is True
    assert result["iterations"] < 100

    for state, expected in exact.items():
        assert result["V"][state] == pytest.approx(expected, abs=1e-10)
        assert result["trace"][state][0] == pytest.approx(0.0)
        assert result["trace"][state][-1] == pytest.approx(expected, abs=1e-10)


def test_partitioned_value_iteration_matches_full_solver_on_piglet_goal_two():
    # Purpose: partitioned iteration is a second algorithmic path; it should
    # agree with full Jacobi iteration on the small exact Piglet case.
    spec = piglet.piglet_make_spec(target_score=2)
    full = vi.value_iteration(spec, tol=1e-12, max_iterations=10_000, restricted_k=True)
    partitioned = vi.partitioned_value_iteration(
        spec,
        tol=1e-12,
        max_local_iterations=10_000,
        restricted_k=True,
    )

    assert partitioned["converged"] is True
    np.testing.assert_allclose(partitioned["V"], full["V"], atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(partitioned["policy"], full["policy"])


def test_extract_policy_marks_invalid_states_and_uses_action_codes():
    # Purpose: plotting and analysis code rely on policy values 1=continue,
    # 0=hold, and -1=invalid.
    spec = piglet.piglet_make_spec(target_score=2)
    result = vi.value_iteration(spec, tol=1e-12, max_iterations=10_000, restricted_k=True)
    policy = result["policy"]
    valid = vi.valid_state_mask(spec, restricted_k=True)

    assert set(np.unique(policy[valid])).issubset({0, 1})
    assert np.all(policy[~valid] == -1)


def test_optimal_policy_function_maps_actual_state_to_solver_state():
    # Purpose: the game-playing adapter must translate actual scores into the
    # current-player perspective expected by the value table.
    spec = piglet.piglet_make_spec(target_score=2)
    result = vi.value_iteration(spec, tol=1e-12, max_iterations=10_000, restricted_k=True)
    policy_fn = vi.optimal_policy_function(spec, result["V"], restricted_k=True)

    assert policy_fn(piglet.PigletState(scores=(0, 0), turn_total=0, current_player=0), None) == "flip"
    assert policy_fn(piglet.PigletState(scores=(1, 0), turn_total=1, current_player=0), None) == "hold"
