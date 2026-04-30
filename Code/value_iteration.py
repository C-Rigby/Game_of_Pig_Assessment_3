"""
value_iteration.py

Value-iteration solver for Piglet and one-die Pig.

This file contains the algorithmic core:

1. State-space construction for the article's state (i, j, k).
2. Bellman action values:
       Q_continue(i,j,k)
       Q_hold(i,j,k)
3. Full Jacobi-style value iteration.
4. partitioned value iteration.
5. Greedy policy extraction.


State convention
----------------
The value-iteration state is always from the current player's perspective:

    (i, j, k)

where

    i = current player's banked score,
    j = opponent's banked score,
    k = current player's unbanked turn total.

For a target score G, the non-terminal states are:

    0 <= i < G,
    0 <= j < G,
    0 <= k < G - i.

If i + k >= G, the current player can hold and win, so the value is 1.
Those states are not treated as unknown equations.

Reward convention
-----------------
This follows the paper's value-iteration interpretation:

    reward = 1 only when the process reaches a winning state,
    reward = 0 otherwise,
    gamma = 1.

Under this convention, V[i,j,k] is the probability that the current player
eventually wins from state (i,j,k), assuming optimal play thereafter.
"""

from __future__ import annotations

from typing import Iterable, Optional
import math
import numpy as np


State = tuple[int, int, int]


# ---------------------------------------------------------------------
# 1. Specification and state-space helpers
# ---------------------------------------------------------------------

def validate_spec(spec: dict) -> None:
    """Validate the minimal game specification needed by the solver.

    Args:
        spec:
            Dictionary returned by piglet.make_spec(...) or pig.make_spec(...).

    Returns:
        None.

    Raises:
        ValueError:
            If required fields are missing or probabilities are invalid.

    Required keys:
        target_score:
            Winning score G.
        continue_action:
            "flip" for Piglet or "roll" for Pig.
        hold_action:
            Usually "hold".
        bust_probability:
            Probability of losing the current turn total.
        gain_outcomes:
            Positive increments of k when the continue action succeeds.
        gain_probabilities:
            Probabilities corresponding to gain_outcomes.
    """

    required = {
        "target_score",
        "continue_action",
        "hold_action",
        "bust_probability",
        "gain_outcomes",
        "gain_probabilities",
    }

    missing = required.difference(spec)
    if missing:
        raise ValueError(f"spec is missing required keys: {sorted(missing)}")

    G = int(spec["target_score"])
    if G < 2:
        raise ValueError("target_score must be at least 2.")

    gain_outcomes = tuple(spec["gain_outcomes"])
    gain_probs = tuple(spec["gain_probabilities"])

    if len(gain_outcomes) != len(gain_probs):
        raise ValueError("gain_outcomes and gain_probabilities must have the same length.")

    if any(int(g) <= 0 for g in gain_outcomes):
        raise ValueError("all gain_outcomes must be positive integers.")

    if any(float(p) < 0 for p in gain_probs):
        raise ValueError("gain probabilities must be non-negative.")

    total_prob = float(spec["bust_probability"]) + float(sum(gain_probs))
    if not math.isclose(total_prob, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"transition probabilities must sum to 1; got {total_prob}.")


def target_score(spec: dict) -> int:
    """Return the target score G.

    Args:
        spec:
            Game specification.

    Returns:
        Integer target score.
    """

    return int(spec["target_score"])


def is_valid_state(spec: dict, i: int, j: int, k: int) -> bool:
    """Check whether (i, j, k) is a valid non-terminal state.

    Args:
        spec:
            Game specification.
        i:
            Current player's banked score.
        j:
            Opponent's banked score.
        k:
            Current turn total.

    Returns:
        True iff

            0 <= i < G,
            0 <= j < G,
            0 <= k < G - i.

    Explanation:
        If i + k >= G, the player can hold and win immediately. Such a state
        has value 1 and is not one of the unknown values solved by iteration.
    """

    G = target_score(spec)
    return 0 <= i < G and 0 <= j < G and 0 <= k < G - i


def iter_states(spec: dict) -> Iterable[State]:
    """Iterate through all valid non-terminal states.

    Args:
        spec:
            Game specification.

    Yields:
        Tuples (i, j, k).
    """

    G = target_score(spec)

    for i in range(G):
        for j in range(G):
            for k in range(G - i):
                yield (i, j, k)


def count_states(spec: dict) -> int:
    """Count valid non-terminal states.

    Args:
        spec:
            Game specification.

    Returns:
        Number of valid states.

    Formula:
        For target score G,

            number of states = G * sum_{i=0}^{G-1} (G-i)
                             = G * G * (G+1) / 2.

        For Pig with G=100, this gives 505000 states.
    """

    G = target_score(spec)
    return G * (G * (G + 1) // 2)


def make_value_table(spec: dict, init_value: float = 0.0) -> np.ndarray:
    """Create a padded value table V.

    Args:
        spec:
            Game specification.
        init_value:
            Initial value assigned to every valid non-terminal state.

    Returns:
        NumPy array V with shape (G, G, G). Valid states are set to init_value;
        invalid entries are set to np.nan.

    Note:
        The padded table is not memory-minimal, but it makes indexing and
        visualization much simpler.
    """

    validate_spec(spec)

    G = target_score(spec)
    V = np.full((G, G, G), np.nan, dtype=float)

    for i, j, k in iter_states(spec):
        V[i, j, k] = float(init_value)

    return V


def valid_state_mask(spec: dict) -> np.ndarray:
    """Create a Boolean mask for valid non-terminal states.

    Args:
        spec:
            Game specification.

    Returns:
        Boolean array with shape (G, G, G). Valid states are True.
    """

    G = target_score(spec)
    mask = np.zeros((G, G, G), dtype=bool)

    for i, j, k in iter_states(spec):
        mask[i, j, k] = True

    return mask


# ---------------------------------------------------------------------
# 2. Bellman equations
# ---------------------------------------------------------------------

def value_at(spec: dict, V: np.ndarray, i: int, j: int, k: int) -> float:
    """Read V[i,j,k] with terminal-state handling.

    Args:
        spec:
            Game specification.
        V:
            Value table.
        i:
            Current player's banked score.
        j:
            Opponent's banked score.
        k:
            Current turn total.

    Returns:
        1.0 if i + k >= G, because holding wins immediately.
        Otherwise returns V[i,j,k].

    Raises:
        ValueError:
            If the requested state is not terminal and not a valid state.
    """

    G = target_score(spec)

    if i + k >= G:
        return 1.0

    if not is_valid_state(spec, i, j, k):
        raise ValueError(f"Invalid non-terminal state: {(i, j, k)}")

    return float(V[i, j, k])


def q_continue(spec: dict, V: np.ndarray, i: int, j: int, k: int) -> float:
    """Compute the value of continuing: flip for Piglet, roll for Pig.

    Args:
        spec:
            Game specification.
        V:
            Current value table.
        i, j, k:
            State coordinates.

    Returns:
        Expected win probability after taking the continue action.

    General formula:
        Let p_b be the bust probability. Let gain outcomes be d with
        probabilities p_d. Then

            Q_continue(i,j,k)
                = p_b * (1 - P[j,i,0])
                  + sum_d p_d * P[i,j,k+d].

    Piglet:
        p_b = 1/2, d = 1, p_d = 1/2.

    Pig:
        p_b = 1/6, d in {2,3,4,5,6}, each with probability 1/6.
    """

    if not is_valid_state(spec, i, j, k):
        raise ValueError(f"Invalid state: {(i, j, k)}")

    bust_prob = float(spec["bust_probability"])
    gains = tuple(int(x) for x in spec["gain_outcomes"])
    gain_probs = tuple(float(x) for x in spec["gain_probabilities"])

    value = bust_prob * (1.0 - value_at(spec, V, j, i, 0))

    for gain, prob in zip(gains, gain_probs):
        value += prob * value_at(spec, V, i, j, k + gain)

    return float(value)


def q_hold(spec: dict, V: np.ndarray, i: int, j: int, k: int) -> float:
    """Compute the value of holding.

    Args:
        spec:
            Game specification.
        V:
            Current value table.
        i, j, k:
            State coordinates.

    Returns:
        Expected win probability after holding.

    Formula:
        Holding banks k points and gives the turn to the opponent. From the
        opponent's perspective, the next state is (j, i+k, 0). Therefore,

            Q_hold(i,j,k) = 1 - P[j,i+k,0].

        If i+k >= G, holding wins immediately and the value is 1.
    """

    if not is_valid_state(spec, i, j, k):
        raise ValueError(f"Invalid state: {(i, j, k)}")

    G = target_score(spec)

    if i + k >= G:
        return 1.0

    return float(1.0 - value_at(spec, V, j, i + k, 0))


def bellman_update(spec: dict, V: np.ndarray, i: int, j: int, k: int) -> float:
    """Apply one Bellman optimality update.

    Args:
        spec:
            Game specification.
        V:
            Current value table.
        i, j, k:
            State coordinates.

    Returns:
        max(Q_continue(i,j,k), Q_hold(i,j,k)).
    """

    return max(
        q_continue(spec, V, i, j, k),
        q_hold(spec, V, i, j, k),
    )


def best_action(
    spec: dict,
    V: np.ndarray,
    i: int,
    j: int,
    k: int,
    tie_action: str = "hold",
) -> str:
    """Return the greedy action under a value table.

    Args:
        spec:
            Game specification.
        V:
            Value table.
        i, j, k:
            State coordinates.
        tie_action:
            Action used when continue and hold have equal values.

    Returns:
        spec["continue_action"] or spec["hold_action"].
    """

    cont = q_continue(spec, V, i, j, k)
    hold = q_hold(spec, V, i, j, k)

    if cont > hold:
        return str(spec["continue_action"])

    if hold > cont:
        return str(spec["hold_action"])

    return tie_action


# ---------------------------------------------------------------------
# 3. Full Jacobi value iteration
# ---------------------------------------------------------------------

def value_iteration(
    spec: dict,
    tol: float = 1e-12,
    max_iterations: int = 100_000,
    init_value: float = 0.0,
    trace_states: Optional[Iterable[State]] = None,
) -> dict:
    """Run full Jacobi-style value iteration over all states.

    Args:
        spec:
            Game specification returned by piglet.make_spec or pig.make_spec.
        tol:
            Convergence tolerance. The algorithm stops when the largest update
            in one full sweep is below tol.
        max_iterations:
            Maximum number of full sweeps.
        init_value:
            Initial value assigned to all valid states.
        trace_states:
            Optional states whose values are recorded after each iteration.
            This is useful for Piglet goal=2 convergence plots.

    Returns:
        Dictionary with keys:
            V:
                Converged value table.
            policy:
                Greedy policy table; 1 means continue, 0 means hold.
            deltas:
                Largest absolute update per iteration.
            trace:
                Dictionary mapping traced states to value sequences.
            iterations:
                Number of completed sweeps.
            converged:
                Whether the tolerance was reached.
            method:
                Method name.
            spec:
                Copy of the game specification.

    Method:
        This is Jacobi-style iteration: every right-hand side is computed from
        old_V, then written into V. It is simple and suitable for Piglet and small
        target-score tests.
    """

    validate_spec(spec)

    V = make_value_table(spec, init_value=init_value)
    states = list(iter_states(spec))

    traced = list(trace_states or [])
    trace = {state: [value_at(spec, V, *state)] for state in traced}

    deltas: list[float] = []

    for iteration in range(1, max_iterations + 1):
        old_V = V.copy()
        delta = 0.0

        for i, j, k in states:
            new_value = bellman_update(spec, old_V, i, j, k)
            delta = max(delta, abs(new_value - old_V[i, j, k]))
            V[i, j, k] = new_value

        deltas.append(delta)

        for state in traced:
            trace[state].append(value_at(spec, V, *state))

        if delta < tol:
            return {
                "V": V,
                "policy": extract_policy(spec, V),
                "deltas": deltas,
                "trace": trace,
                "iterations": iteration,
                "converged": True,
                "method": "full_jacobi_value_iteration",
                "spec": dict(spec),
            }

    return {
        "V": V,
        "policy": extract_policy(spec, V),
        "deltas": deltas,
        "trace": trace,
        "iterations": max_iterations,
        "converged": False,
        "method": "full_jacobi_value_iteration",
        "spec": dict(spec),
    }


# ---------------------------------------------------------------------
# 4.  partitioned value iteration
# ---------------------------------------------------------------------

def _local_value_at(
    spec: dict,
    V: np.ndarray,
    old_local: dict[State, float],
    i: int,
    j: int,
    k: int,
) -> float:
    """Read a value during local subpartition iteration.

    Args:
        spec:
            Game specification.
        V:
            Global value table.
        old_local:
            Previous local values for states in the current subpartition.
        i, j, k:
            State coordinates.

    Returns:
        Terminal value 1 if i+k >= G; otherwise the old local value if the
        state is in the current subpartition; otherwise the fixed global value.
    """

    G = target_score(spec)

    if i + k >= G:
        return 1.0

    if not is_valid_state(spec, i, j, k):
        raise ValueError(f"Invalid non-terminal state: {(i, j, k)}")

    return float(old_local.get((i, j, k), V[i, j, k]))


def _local_q_continue(
    spec: dict,
    V: np.ndarray,
    old_local: dict[State, float],
    i: int,
    j: int,
    k: int,
) -> float:
    """Local-Jacobi version of q_continue.

    Args:
        spec:
            Game specification.
        V:
            Global value table.
        old_local:
            Previous local values for the current subpartition.
        i, j, k:
            State coordinates.

    Returns:
        Continue action value using old_local whenever possible.
    """

    bust_prob = float(spec["bust_probability"])
    gains = tuple(int(x) for x in spec["gain_outcomes"])
    gain_probs = tuple(float(x) for x in spec["gain_probabilities"])

    value = bust_prob * (1.0 - _local_value_at(spec, V, old_local, j, i, 0))

    for gain, prob in zip(gains, gain_probs):
        value += prob * _local_value_at(spec, V, old_local, i, j, k + gain)

    return float(value)


def _local_q_hold(
    spec: dict,
    V: np.ndarray,
    old_local: dict[State, float],
    i: int,
    j: int,
    k: int,
) -> float:
    """Local-Jacobi version of q_hold.

    Args:
        spec:
            Game specification.
        V:
            Global value table.
        old_local:
            Previous local values for the current subpartition.
        i, j, k:
            State coordinates.

    Returns:
        Hold action value using old_local whenever possible.
    """

    G = target_score(spec)

    if i + k >= G:
        return 1.0

    return float(1.0 - _local_value_at(spec, V, old_local, j, i + k, 0))


def _score_sum_pairs(G: int, score_sum: int) -> list[tuple[int, int]]:
    """Return unordered score pairs with a fixed score sum.

    Args:
        G:
            Target score.
        score_sum:
            Desired i+j.

    Returns:
        List of pairs (i,j) such that 0 <= i <= j < G and i+j=score_sum.
    """

    pairs: list[tuple[int, int]] = []

    i_min = max(0, score_sum - (G - 1))
    i_max = min(G - 1, score_sum)

    for i in range(i_min, i_max + 1):
        j = score_sum - i

        if 0 <= j < G and i <= j:
            pairs.append((i, j))

    return pairs


def _subpartition_states(spec: dict, i: int, j: int) -> list[State]:
    """Return states updated together for one local subpartition.

    Args:
        spec:
            Game specification.
        i:
            First score.
        j:
            Second score.

    Returns:
        All states (i,j,k) and, if i != j, all states (j,i,k).

    Rationale:
        In a fixed score-sum partition, dependencies are either already solved
        greater-score-sum states, or states with the player scores switched.
        Therefore (i,j,*) and (j,i,*) should be locally iterated together.
    """

    G = target_score(spec)

    states: list[State] = [(i, j, k) for k in range(G - i)]

    if i != j:
        states.extend((j, i, k) for k in range(G - j))

    return states


def partitioned_value_iteration(
    spec: dict,
    tol: float = 1e-12,
    max_local_iterations: int = 100_000,
    init_value: float = 0.0,
    progress: bool = False,
) -> dict:
    """Run article-aligned partitioned value iteration.

    Args:
        spec:
            Game specification. This is mainly intended for Pig with target 100,
            but it also works for Piglet or smaller Pig tests.
        tol:
            Local subpartition convergence tolerance.
        max_local_iterations:
            Maximum iterations per local subpartition.
        init_value:
            Initial value for all valid states.
        progress:
            If True, print score-sum progress.

    Returns:
        Dictionary with keys:
            V:
                Value table of optimal win probabilities.
            policy:
                Greedy policy table; 1 means continue, 0 means hold.
            local_iterations:
                Mapping from score pair (i,j) to local iteration count.
            max_local_delta:
                Final local delta for each score pair.
            converged:
                Whether all subpartitions converged.
            method:
                Method name.
            spec:
                Copy of the game specification.

    Article-aligned acceleration:
        Player scores never decrease, so the score sum i+j never decreases.
        Hence probabilities with a given score sum do not depend on lower
        score sums. The algorithm therefore solves score sums in descending
        order. Inside each score-sum level, it jointly iterates the two orientations
        (i,j,k) and (j,i,k).
    """

    validate_spec(spec)

    G = target_score(spec)
    V = make_value_table(spec, init_value=init_value)

    local_iterations: dict[tuple[int, int], int] = {}
    max_local_delta: dict[tuple[int, int], float] = {}
    all_converged = True

    for score_sum in range(2 * G - 2, -1, -1):
        if progress:
            print(f"Solving score_sum={score_sum}")

        for i, j in _score_sum_pairs(G, score_sum):
            states = _subpartition_states(spec, i, j)

            final_delta = math.inf
            converged = False

            for iteration in range(1, max_local_iterations + 1):
                old_local = {state: float(V[state]) for state in states}
                new_values: dict[State, float] = {}
                delta = 0.0

                for state in states:
                    a, b, k = state

                    cont = _local_q_continue(spec, V, old_local, a, b, k)
                    hold = _local_q_hold(spec, V, old_local, a, b, k)
                    new_value = max(cont, hold)

                    new_values[state] = new_value
                    delta = max(delta, abs(new_value - old_local[state]))

                for state, new_value in new_values.items():
                    V[state] = new_value

                final_delta = delta

                if delta < tol:
                    converged = True
                    break

            local_iterations[(i, j)] = iteration
            max_local_delta[(i, j)] = final_delta

            if not converged:
                all_converged = False

    return {
        "V": V,
        "policy": extract_policy(spec, V),
        "local_iterations": local_iterations,
        "max_local_delta": max_local_delta,
        "converged": all_converged,
        "method": "partitioned_score_sum_jacobi_value_iteration",
        "spec": dict(spec),
    }


# ---------------------------------------------------------------------
# 5. Policy extraction
# ---------------------------------------------------------------------

def extract_policy(
    spec: dict,
    V: np.ndarray,
    tie_continue: bool = False,
) -> np.ndarray:
    """Extract the greedy optimal policy from a value table.

    Args:
        spec:
            Game specification.
        V:
            Value table.
        tie_continue:
            If True, exact ties choose continue; otherwise ties choose hold.

    Returns:
        Integer array with shape (G,G,G):
            1  = continue action, i.e. flip or roll;
            0  = hold;
            -1 = invalid state.
    """

    G = target_score(spec)
    policy = np.full((G, G, G), -1, dtype=np.int8)

    for i, j, k in iter_states(spec):
        cont = q_continue(spec, V, i, j, k)
        hold = q_hold(spec, V, i, j, k)

        if cont > hold or (
            tie_continue and math.isclose(cont, hold, rel_tol=0.0, abs_tol=1e-15)
        ):
            policy[i, j, k] = 1
        else:
            policy[i, j, k] = 0

    return policy


def action_value_tables(spec: dict, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute full Q_continue and Q_hold tables.

    Args:
        spec:
            Game specification.
        V:
            Value table.

    Returns:
        Pair (Q_continue, Q_hold), each with shape (G,G,G). Invalid states
        are np.nan.
    """

    G = target_score(spec)

    Qc = np.full((G, G, G), np.nan, dtype=float)
    Qh = np.full((G, G, G), np.nan, dtype=float)

    for i, j, k in iter_states(spec):
        Qc[i, j, k] = q_continue(spec, V, i, j, k)
        Qh[i, j, k] = q_hold(spec, V, i, j, k)

    return Qc, Qh


def optimal_policy_function(spec: dict, V: np.ndarray):
    """Create a game-playing policy function from a solved value table.

    Args:
        spec:
            Game specification.
        V:
            Solved value table.

    Returns:
        Function policy(state, rng) compatible with pig.play_game or
        piglet.play_game.

    Input-state requirement:
        The state object should have:
            scores,
            turn_total,
            current_player.

    The function converts the actual game state into the current-player
    perspective (i,j,k), then selects the greedy action.
    """

    policy_table = extract_policy(spec, V)

    continue_action = str(spec["continue_action"])
    hold_action = str(spec["hold_action"])
    G = target_score(spec)

    def policy(state, rng):
        p = state.current_player
        q = 1 - p

        i = state.scores[p]
        j = state.scores[q]
        k = state.turn_total

        if i + k >= G:
            return hold_action

        if not is_valid_state(spec, i, j, k):
            raise ValueError(f"State cannot be represented as (i,j,k): {(i,j,k)}")

        return continue_action if policy_table[i, j, k] == 1 else hold_action

    return policy