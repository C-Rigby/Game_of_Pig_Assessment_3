"""
analysis_helpers.py

Analysis, validation, reachability, and plotting helpers for the Pig/Piglet
value-iteration project.

This file does not solve the value-iteration equations directly. Instead, it
uses outputs from value_iteration.py to:

1. Validate Piglet goal=2 against the exact values reported in the paper.
2. Extract data for article-style figures.
3. Compute roll/hold or flip/hold boundaries.
4. Compute reachable states under an optimal policy.
5. Plot figures corresponding to the paper's visual analyses.

Expected inputs
---------------
Most functions take:

    spec:
        Dictionary returned by piglet.make_spec(...) or pig.make_spec(...).

    V:
        Value table returned by value_iteration.value_iteration(...) or
        value_iteration.partitioned_value_iteration(...).

    policy:
        Policy table returned by value_iteration.extract_policy(...), where:
            1 = continue action, i.e. flip or roll,
            0 = hold,
           -1 = invalid state.

Plotting dependencies
---------------------
The plotting functions use matplotlib. The probability contour function can
use skimage.measure.marching_cubes if scikit-image is installed. If scikit-image
is not available, a simple point-cloud fallback is used.

The goal is not to copy the paper's exact old rendering style pixel-for-pixel,
but to reproduce the same mathematical visual objects:

    Figure 2: Piglet value-iteration convergence.
    Figure 3: Pig roll/hold boundary.
    Figure 4: Cross-section at opponent score j=30.
    Figure 5: Reachable states for an optimal player.
    Figure 6: Reachable states where continuing is optimal.
    Figure 7: Win-probability contours.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional
import math
import numpy as np

from . import value_iteration as vi


State = tuple[int, int, int]
ActualState = tuple[int, int, int, int]


# ---------------------------------------------------------------------
# 1. Piglet goal=2 validation and trace helpers
# ---------------------------------------------------------------------

def piglet_goal2_trace_states() -> list[State]:
    """Return the six Piglet goal-2 states used for convergence tracing.

    Args:
        None.

    Returns:
        List of states:
            (0,0,0), (0,0,1), (0,1,0),
            (0,1,1), (1,0,0), (1,1,0).

    Use:
        Pass this list to value_iteration.value_iteration(..., trace_states=...).
    """

    return [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
    ]


def piglet_goal2_exact_values() -> dict[State, float]:
    """Return exact Piglet goal=2 probabilities from the paper.

    Args:
        None.

    Returns:
        Dictionary mapping state (i,j,k) to exact win probability.

    Exact values:
        P_0,0,0 = 4/7
        P_0,0,1 = 5/7
        P_0,1,0 = 2/5
        P_0,1,1 = 3/5
        P_1,0,0 = 4/5
        P_1,1,0 = 2/3
    """

    return {
        (0, 0, 0): 4.0 / 7.0,
        (0, 0, 1): 5.0 / 7.0,
        (0, 1, 0): 2.0 / 5.0,
        (0, 1, 1): 3.0 / 5.0,
        (1, 0, 0): 4.0 / 5.0,
        (1, 1, 0): 2.0 / 3.0,
    }


def check_piglet_goal2_solution(
    V: np.ndarray,
    atol: float = 1e-10,
) -> dict[State, float]:
    """Check a solved Piglet goal=2 value table against exact values.

    Args:
        V:
            Value table for Piglet with target_score=2.
        atol:
            Absolute tolerance.

    Returns:
        Dictionary mapping each checked state to its absolute error.

    Raises:
        AssertionError:
            If any absolute error exceeds atol.
    """

    errors: dict[State, float] = {}

    for state, exact in piglet_goal2_exact_values().items():
        got = float(V[state])
        err = abs(got - exact)
        errors[state] = err

        if err > atol:
            raise AssertionError(
                f"State {state}: got {got}, expected {exact}, error {err}"
            )

    return errors


def trace_to_table(trace: dict[State, list[float]]) -> dict[str, list[float]]:
    """Convert a trace dictionary into a simple table-like dictionary.

    Args:
        trace:
            Dictionary returned in result["trace"] by value_iteration(...).

    Returns:
        Dictionary with one key "iteration" and one key per traced state.

    Example output keys:
        "iteration"
        "P_0_0_0"
        "P_0_0_1"
    """

    if not trace:
        return {"iteration": []}

    n = len(next(iter(trace.values())))
    table: dict[str, list[float]] = {"iteration": list(range(n))}

    for state, values in trace.items():
        i, j, k = state
        table[f"P_{i}_{j}_{k}"] = list(values)

    return table


# ---------------------------------------------------------------------
# 2. Boundary and action-value data
# ---------------------------------------------------------------------

def hold_boundary(policy: np.ndarray) -> np.ndarray:
    """Find the first k where holding is optimal for each score pair.

    Args:
        policy:
            Policy table where 1 means continue and 0 means hold.

    Returns:
        Float array B with shape (G,G). B[i,j] is the smallest valid k such
        that policy[i,j,k] == 0. If no hold action exists, B[i,j] is np.nan.

    Use:
        This is the main data object for Figure 3 and Figure 4.
    """

    G = policy.shape[0]
    boundary = np.full((G, G), np.nan, dtype=float)

    for i in range(G):
        for j in range(G):
            valid_k = np.where(policy[i, j, :] >= 0)[0]
            if valid_k.size == 0:
                continue

            hold_k = valid_k[policy[i, j, valid_k] == 0]
            if hold_k.size > 0:
                boundary[i, j] = float(hold_k[0])

    return boundary


def boundary_points(policy: np.ndarray) -> np.ndarray:
    """Return roll/hold boundary as an array of 3D points.

    Args:
        policy:
            Policy table.

    Returns:
        Array with shape (n_points, 3), where each row is (i,j,k_boundary).

    Notes:
        Rows with no hold boundary are omitted.
    """

    B = hold_boundary(policy)
    points: list[tuple[float, float, float]] = []

    G = B.shape[0]

    for i in range(G):
        for j in range(G):
            if not np.isnan(B[i, j]):
                points.append((float(i), float(j), float(B[i, j])))

    return np.asarray(points, dtype=float)


def _split_point_segments(points: np.ndarray) -> list[np.ndarray]:
    """Split sorted ``(i,k)`` points where adjacent player scores are not contiguous."""

    if points.size == 0:
        return []

    segments: list[np.ndarray] = []
    start = 0

    for idx in range(1, points.shape[0]):
        if int(points[idx, 0]) != int(points[idx - 1, 0]) + 1:
            segments.append(points[start:idx])
            start = idx

    segments.append(points[start:])

    return segments


def _cross_section_transition_segments(
    policy: np.ndarray,
    opponent_score: int,
) -> tuple[np.ndarray, list[dict[str, np.ndarray | int]]]:
    """Return all action-change boundaries in a fixed-opponent-score slice.

    The older ``hold_boundary`` helper records only the first ``roll -> hold``
    transition for each score pair. Figure 4 in the paper needs the full
    cross-section: some columns change action more than once.
    """

    G = policy.shape[0]
    transition_rows: list[tuple[float, float, float, float, float]] = []
    transitions_by_order: dict[int, list[tuple[float, float]]] = {}

    for i in range(G):
        valid_k = np.where(policy[i, opponent_score, :] >= 0)[0]
        if valid_k.size < 2:
            continue

        transition_order = 0
        previous_k = int(valid_k[0])
        previous_action = int(policy[i, opponent_score, previous_k])

        for k_value in valid_k[1:]:
            k = int(k_value)
            action = int(policy[i, opponent_score, k])

            if action != previous_action:
                transition_order += 1
                row = (
                    float(i),
                    float(k),
                    float(previous_action),
                    float(action),
                    float(transition_order),
                )
                transition_rows.append(row)
                transitions_by_order.setdefault(transition_order, []).append(
                    (float(i), float(k))
                )

            previous_action = action

    segments: list[dict[str, np.ndarray | int]] = []

    for transition_order, points in transitions_by_order.items():
        ordered = np.asarray(points, dtype=float)
        ordered = ordered[np.argsort(ordered[:, 0])]

        for segment in _split_point_segments(ordered):
            segments.append(
                {
                    "points": segment,
                    "transition_order": transition_order,
                }
            )

    if transition_rows:
        transition_points = np.asarray(transition_rows, dtype=float)
    else:
        transition_points = np.empty((0, 5), dtype=float)

    return transition_points, segments


def _terminal_boundary_segment(policy: np.ndarray) -> np.ndarray:
    """Return the ``i + k = G`` boundary visible in unrestricted Pig figures."""

    G = policy.shape[0]
    points = [
        (float(i), float(G - i))
        for i in range(1, G + 1)
        if 0 <= G - i < G
    ]

    return np.asarray(points, dtype=float)


def _standard_pig_spec_for_policy(policy: np.ndarray) -> dict:
    """Create the minimal Pig spec needed for Figure 4 reachability shading."""

    G = int(policy.shape[0])

    return {
        "name": f"pig_goal_{G}",
        "game": "pig",
        "target_score": G,
        "continue_action": "roll",
        "hold_action": "hold",
        "bust_probability": 1.0 / 6.0,
        "gain_outcomes": (2, 3, 4, 5, 6),
        "gain_probabilities": (1.0 / 6.0,) * 5,
    }


def cross_section_boundary(
    policy: np.ndarray,
    opponent_score: int = 30,
    spec: Optional[dict] = None,
    reachable: Optional[np.ndarray] = None,
    restricted_k: Optional[bool] = None,
    opponent_mode: str = "any",
    include_reachable: bool = True,
) -> dict[str, object]:
    """Extract a fixed-opponent-score boundary section.

    Args:
        policy:
            Policy table.
        opponent_score:
            Fixed value of j. The paper uses j=30 in Figure 4.
        spec:
            Optional game specification. If omitted, a standard one-die Pig
            spec is inferred from ``policy.shape[0]`` for Figure 4 shading.
        reachable:
            Optional precomputed reachable mask with shape ``(G,G,G)``.
        restricted_k:
            Whether reachability should use the restricted state space. If
            omitted, this is inferred from whether ``policy`` contains invalid
            entries.
        opponent_mode:
            Passed to ``reachable_states_for_optimal_player`` when reachable
            shading is computed here.
        include_reachable:
            Whether to include the Figure 4 reachable-state cross-section.

    Returns:
        Dictionary with:
            player_score:
                Array of i values.
            boundary_k:
                First k where holding is optimal for each i. This is retained
                for backwards compatibility.
            transition_points:
                Rows ``(i, k, from_action, to_action, transition_order)`` for
                every action change in the cross-section.
            boundary_segments:
                Contiguous action-change segments ready for plotting.
            terminal_boundary:
                Points on ``i + k = G``. This is the upper diagonal visible in
                the paper's unrestricted Figure 4.
            hold_mask:
                Boolean ``(i,k)`` mask for hold states in this cross-section.
                The paper-style plot uses this to draw continuous region
                contours instead of disconnected transition samples.
            reachable_mask:
                Boolean ``(i,k)`` mask for the selected opponent score, or
                ``None`` if reachability was not requested.
            opponent_score:
                The fixed j value.
    """

    B = hold_boundary(policy)
    G = B.shape[0]

    if not 0 <= opponent_score < G:
        raise ValueError(f"opponent_score must be in [0,{G-1}]")

    transition_points, boundary_segments = _cross_section_transition_segments(
        policy,
        opponent_score,
    )

    if restricted_k is None:
        restricted_k = bool(np.any(policy < 0))

    reachable_mask = None

    if include_reachable:
        if reachable is None:
            reach_spec = spec if spec is not None else _standard_pig_spec_for_policy(policy)
            reachable = reachable_states_for_optimal_player(
                reach_spec,
                policy,
                restricted_k=restricted_k,
                opponent_mode=opponent_mode,
            )

        reachable_mask = reachable[:, opponent_score, :]

    return {
        "player_score": np.arange(G),
        "boundary_k": B[:, opponent_score],
        "transition_points": transition_points,
        "boundary_segments": boundary_segments,
        "terminal_boundary": _terminal_boundary_segment(policy),
        "hold_mask": policy[:, opponent_score, :] == 0,
        "reachable_mask": reachable_mask,
        "opponent_score": opponent_score,
    }


def continue_mask(policy: np.ndarray) -> np.ndarray:
    """Return a Boolean mask for states where continuing is optimal.

    Args:
        policy:
            Policy table.

    Returns:
        Boolean mask with True where policy == 1.
    """

    return policy == 1


def hold_mask(policy: np.ndarray) -> np.ndarray:
    """Return a Boolean mask for states where holding is optimal.

    Args:
        policy:
            Policy table.

    Returns:
        Boolean mask with True where policy == 0.
    """

    return policy == 0


# ---------------------------------------------------------------------
# 3. Reachability analysis
# ---------------------------------------------------------------------

def _actual_to_value_state(actual: ActualState) -> State:
    """Convert actual player-0-turn state into value-iteration state.

    Args:
        actual:
            Tuple (score0, score1, turn_total, current_player).

    Returns:
        State (i,j,k) from player 0's perspective.

    Raises:
        ValueError:
            If current_player is not 0.
    """

    score0, score1, turn_total, current_player = actual

    if current_player != 0:
        raise ValueError("This conversion is only for player 0's turn.")

    return (score0, score1, turn_total)


def _expand_actual_state(
    spec: dict,
    policy: np.ndarray,
    actual: ActualState,
    opponent_mode: str,
    restricted_k: bool,
) -> list[ActualState]:
    """Expand one actual game state for reachability analysis.

    Args:
        spec:
            Game specification.
        policy:
            Optimal policy table for player 0.
        actual:
            Actual game state (score0, score1, turn_total, current_player).
        opponent_mode:
            "any" means the opponent may choose any legal action.
            "same_policy" means the opponent also follows the policy.

    Returns:
        List of next actual states. Terminal outcomes are omitted.
    """

    G = vi.target_score(spec)
    gains = tuple(int(x) for x in spec["gain_outcomes"])

    score0, score1, k, player = actual
    next_states: list[ActualState] = []

    if player == 0:
        if score0 + k >= G:
            return []

        if not vi.is_valid_state(spec, score0, score1, k, restricted_k = restricted_k):
            return []

        action_code = int(policy[score0, score1, k])
        actions = ["continue"] if action_code == 1 else ["hold"]

    else:
        if score1 + k >= G:
            return []

        if opponent_mode == "any":
            actions = ["continue", "hold"]
        elif opponent_mode == "same_policy":
            if not vi.is_valid_state(spec, score1, score0, k):
                return []
            action_code = int(policy[score1, score0, k])
            actions = ["continue"] if action_code == 1 else ["hold"]
        else:
            raise ValueError("opponent_mode must be 'any' or 'same_policy'.")

    for action in actions:
        if action == "hold":
            if player == 0:
                new_score0 = score0 + k
                if new_score0 < G:
                    next_states.append((new_score0, score1, 0, 1))
            else:
                new_score1 = score1 + k
                if new_score1 < G:
                    next_states.append((score0, new_score1, 0, 0))

        else:
            # Bust outcome.
            if player == 0:
                next_states.append((score0, score1, 0, 1))
            else:
                next_states.append((score0, score1, 0, 0))

            # Positive gain outcomes.
            for gain in gains:
                new_k = k + gain

                if player == 0:
                    if score0 + new_k < G:
                        next_states.append((score0, score1, new_k, 0))
                else:
                    if score1 + new_k < G:
                        next_states.append((score0, score1, new_k, 1))

    return next_states


def reachable_states_for_optimal_player(
    spec: dict,
    policy: np.ndarray,
    restricted_k: bool,
    opponent_mode: str = "any",
) -> np.ndarray:
    """Compute states reachable for player 0 when player 0 follows policy.

    Args:
        spec:
            Game specification.
        policy:
            Policy table where 1 means continue and 0 means hold.
        opponent_mode:
            "any":
                Opponent may take either legal action. This is closest to the
                paper's statement that the optimal player's reachable states are
                reachable regardless of the opponent's policy.
            "same_policy":
                Both players follow the same optimal policy.

    Returns:
        Boolean array with shape (G,G,G). A True entry at (i,j,k) means that
        player 0 can be to act with score i, opponent score j, and turn total k.

    Note:
        This function records only player-0-turn states, because the paper's
        figures are from one player's perspective.
    """

    G = vi.target_score(spec)
    reachable = np.zeros((G, G, G), dtype=bool)

    start: ActualState = (0, 0, 0, 0)
    queue: deque[ActualState] = deque([start])
    seen_actual: set[ActualState] = {start}

    while queue:
        actual = queue.popleft()
        score0, score1, k, player = actual

        if player == 0 and vi.is_valid_state(spec, score0, score1, k, restricted_k = restricted_k):
            reachable[score0, score1, k] = True

        for nxt in _expand_actual_state(spec, policy, actual, opponent_mode, restricted_k = restricted_k):
            s0, s1, tk, p = nxt

            # Keep only bounded non-terminal actual states.
            if not (0 <= s0 < G and 0 <= s1 < G and 0 <= tk < G):
                continue

            if nxt not in seen_actual:
                seen_actual.add(nxt)
                queue.append(nxt)

    return reachable


def reachable_continue_mask(policy: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    """Return reachable states where continuing is optimal.

    Args:
        policy:
            Policy table.
        reachable:
            Reachability mask.

    Returns:
        Boolean mask where reachable is True and policy == 1.
    """

    return np.logical_and(reachable, policy == 1)


def mask_to_points(mask: np.ndarray) -> np.ndarray:
    """Convert a 3D Boolean mask into 3D point coordinates.

    Args:
        mask:
            Boolean array with shape (G,G,G).

    Returns:
        Array with shape (n_points, 3), with rows (i,j,k).
    """

    coords = np.argwhere(mask)
    return coords.astype(float)


# ---------------------------------------------------------------------
# 4. Figure-data extraction
# ---------------------------------------------------------------------

def figure2_data_from_result(result: dict) -> dict[str, list[float]]:
    """Extract Figure 2 data from a Piglet value-iteration result.

    Args:
        result:
            Output dictionary from value_iteration.value_iteration(...).

    Returns:
        Table-like dictionary with iteration and probability sequences.

    Requirement:
        result["trace"] should have been generated by passing
        trace_states=piglet_goal2_trace_states().
    """

    return trace_to_table(result.get("trace", {}))


def figure3_boundary_data(policy: np.ndarray) -> dict[str, np.ndarray]:
    """Extract Figure 3 roll/hold boundary data.

    Args:
        policy:
            Policy table.

    Returns:
        Dictionary with:
            I:
                Meshgrid of current-player scores.
            J:
                Meshgrid of opponent scores.
            K:
                Boundary turn total surface.
            points:
                Boundary points as rows (i,j,k).
    """

    B = hold_boundary(policy)
    G = B.shape[0]
    I, J = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")

    return {
        "I": I,
        "J": J,
        "K": B,
        "points": boundary_points(policy),
    }


def figure4_cross_section_data(
    policy: np.ndarray,
    opponent_score: int = 30,
    spec: Optional[dict] = None,
    reachable: Optional[np.ndarray] = None,
    restricted_k: Optional[bool] = None,
    opponent_mode: str = "any",
    include_reachable: bool = True,
) -> dict[str, object]:
    """Extract Figure 4 fixed-opponent-score cross-section data.

    Args:
        policy:
            Policy table.
        opponent_score:
            Fixed opponent score j. The paper uses j=30.
        spec:
            Optional game specification used for reachable-state shading.
        reachable:
            Optional precomputed reachable mask.
        restricted_k:
            Whether to use the restricted state space for reachability. If
            omitted, this is inferred from the policy table.
        opponent_mode:
            Opponent reachability mode.
        include_reachable:
            Whether to include the reachable-state mask used by the paper-style
            Figure 4 plot.

    Returns:
        Dictionary containing player scores and boundary k values.
    """

    return cross_section_boundary(
        policy,
        opponent_score=opponent_score,
        spec=spec,
        reachable=reachable,
        restricted_k=restricted_k,
        opponent_mode=opponent_mode,
        include_reachable=include_reachable,
    )




def figure5_reachable_data(
    spec: dict,
    policy: np.ndarray,
    restricted_k: bool,
    opponent_mode: str = "any",
) -> dict[str, np.ndarray]:
    """Extract Figure 5 reachable-state data.

    Args:
        spec:
            Game specification.
        policy:
            Policy table.
        opponent_mode:
            "any" or "same_policy".

    Returns:
        Dictionary with:
            reachable:
                Boolean reachable mask.
            points:
                Reachable points as rows (i,j,k).
    """

    reachable = reachable_states_for_optimal_player(
        spec,
        policy,
        opponent_mode=opponent_mode,
        restricted_k = restricted_k,
    )

    return {
        "reachable": reachable,
        "points": mask_to_points(reachable),
    }


def figure6_reachable_continue_data(
    spec: dict,
    policy: np.ndarray,
    restricted_k: bool,
    opponent_mode: str = "any",
) -> dict[str, np.ndarray]:
    """Extract Figure 6 reachable-continue-state data.

    Args:
        spec:
            Game specification.
        policy:
            Policy table.
        opponent_mode:
            "any" or "same_policy".

    Returns:
        Dictionary with:
            reachable:
                Boolean reachable mask.
            reachable_continue:
                Boolean mask of reachable states where continuing is optimal.
            points:
                Points for reachable-continue states.
    """

    reachable = reachable_states_for_optimal_player(
        spec,
        policy,
        opponent_mode=opponent_mode,
        restricted_k = restricted_k,
    )
    rc = reachable_continue_mask(policy, reachable)

    return {
        "reachable": reachable,
        "reachable_continue": rc,
        "points": mask_to_points(rc),
    }


def figure7_probability_contour_data(
    spec: dict,
    V: np.ndarray,
    restricted_k: bool,
    levels: tuple[float, ...] = (0.03, 0.09, 0.27, 0.81),
) -> dict[str, object]:
    """Prepare win-probability contour data.

    Args:
        spec:
            Game specification.
        V:
            Solved value table.
        levels:
            Probability contour levels. The paper uses 3%, 9%, 27%, and 81%.

    Returns:
        Dictionary with:
            V_filled:
                Value table with invalid entries filled by np.nan.
            levels:
                Contour levels.
            valid_mask:
                Valid-state mask.
    """

    return {
        "V_filled": np.array(V, copy=True),
        "levels": tuple(float(x) for x in levels),
        "valid_mask": vi.valid_state_mask(spec, restricted_k = restricted_k),
    }


# ---------------------------------------------------------------------
# 5. Plotting helpers
# ---------------------------------------------------------------------

def _get_pyplot():
    """Import matplotlib.pyplot lazily.

    Args:
        None.

    Returns:
        matplotlib.pyplot module.

    Raises:
        ImportError:
            If matplotlib is not installed.
    """

    import matplotlib.pyplot as plt
    return plt


def plot_figure2_piglet_trace(
    figure2_data: dict[str, list[float]],
    ax=None,
    title: str = "Value Iteration with Piglet (goal points = 2)",
):
    """Plot Piglet value-iteration convergence curves.

    Args:
        figure2_data:
            Output from figure2_data_from_result(...).
        ax:
            Optional matplotlib axis.
        title:
            Plot title.

    Returns:
        Matplotlib axis.
    """

    plt = _get_pyplot()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    iterations = figure2_data.get("iteration", [])

    for key, values in figure2_data.items():
        if key == "iteration":
            continue
        ax.plot(iterations, values, label=key)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Probability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_figure3_policy_boundary(
    boundary_data: dict[str, np.ndarray],
    ax=None,
    title: str = "Roll/Hold Boundary for Optimal Pig Policy",
):
    """Plot a 3D roll/hold boundary surface.

    Args:
        boundary_data:
            Output from figure3_boundary_data(...).
        ax:
            Optional 3D matplotlib axis.
        title:
            Plot title.

    Returns:
        Matplotlib 3D axis.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    I = boundary_data["I"]
    J = boundary_data["J"]
    K = boundary_data["K"]

    ax.plot_surface(I, J, K, linewidth=0, alpha=0.75)
    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Player 2 Score (j)")
    ax.set_zlabel("Turn Total (k)")
    ax.set_title(title)

    return ax


def plot_figure4_cross_section(
    cross_data: dict[str, object],
    ax=None,
    hold_at_threshold: int = 20,
    title: Optional[str] = None,
):
    """Plot fixed-opponent-score roll/hold boundary.

    Args:
        cross_data:
            Output from figure4_cross_section_data(...).
        ax:
            Optional matplotlib axis.
        hold_at_threshold:
            Baseline threshold line, usually 20 for the full Pig game.
        title:
            Optional plot title.

    Returns:
        Matplotlib axis.

    Notes:
        The paper's Figure 4 shows the contour of the roll/hold regions. When
        ``hold_mask`` is present, this function draws that continuous contour
        instead of connecting raw transition samples.
    """

    plt = _get_pyplot()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    i = np.asarray(cross_data["player_score"])
    k = np.asarray(cross_data["boundary_k"])
    opponent_score = int(cross_data["opponent_score"])
    G = int(i.size)

    reachable_mask = cross_data.get("reachable_mask")
    drew_reachable = False

    if reachable_mask is not None:
        reachable_array = np.asarray(reachable_mask, dtype=bool)

        if reachable_array.any():
            ax.contourf(
                np.arange(reachable_array.shape[0]),
                np.arange(reachable_array.shape[1]),
                reachable_array.T.astype(float),
                levels=[0.5, 1.5],
                colors=["#d9d9d9"],
                alpha=0.75,
                zorder=0,
            )
            drew_reachable = True

    boundary_label_used = False
    drew_contour_boundary = False
    hold_mask = cross_data.get("hold_mask")

    if hold_mask is not None:
        hold_array = np.asarray(hold_mask, dtype=bool)

        if hold_array.any() and np.logical_not(hold_array).any():
            ax.contour(
                np.arange(hold_array.shape[0]),
                np.arange(hold_array.shape[1]),
                hold_array.T.astype(float),
                levels=[0.5],
                colors=["#333333"],
                linewidths=2.5,
                zorder=3,
            )
            boundary_label_used = True
            drew_contour_boundary = True

    if not drew_contour_boundary:
        boundary_segments = cross_data.get("boundary_segments", [])

        for segment_info in boundary_segments:
            points = np.asarray(segment_info["points"], dtype=float)
            if points.size == 0:
                continue

            label = "Optimal Boundary" if not boundary_label_used else None
            boundary_label_used = True

            if points.shape[0] == 1:
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    marker="o",
                    markersize=3,
                    linestyle="None",
                    color="#333333",
                    label=label,
                    zorder=3,
                )
            else:
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    color="#333333",
                    linewidth=2.5,
                    solid_capstyle="round",
                    label=label,
                    zorder=3,
                )

        terminal_boundary = cross_data.get("terminal_boundary")
        if terminal_boundary is not None:
            terminal_points = np.asarray(terminal_boundary, dtype=float)

            if terminal_points.size:
                visible = terminal_points[:, 1] <= 50
                terminal_points = terminal_points[visible]

            if terminal_points.size:
                label = "Optimal Boundary" if not boundary_label_used else None
                boundary_label_used = True
                ax.plot(
                    terminal_points[:, 0],
                    terminal_points[:, 1],
                    color="#333333",
                    linewidth=2.5,
                    solid_capstyle="round",
                    label=label,
                    zorder=3,
                )

    if not boundary_label_used:
        # No finite hold boundary exists for this cross-section.
        # This can happen in small demo games or poorly chosen opponent_score.
        ax.plot([], [], label="No finite hold boundary in this section")

    hold_line = ax.axhline(
        hold_at_threshold,
        linestyle="--",
        linewidth=1.2,
        color="#9e9e9e",
        label=f"Hold at {hold_at_threshold}",
        zorder=2,
    )

    from matplotlib.lines import Line2D

    legend_handles = []
    legend_labels = []

    if boundary_label_used:
        legend_handles.append(
            Line2D([0], [0], color="#333333", linewidth=2.5)
        )
        legend_labels.append("Optimal Boundary")

    if drew_reachable:
        from matplotlib.patches import Patch

        legend_handles.append(
            Patch(facecolor="#d9d9d9", edgecolor="#bfbfbf", label="Reachable"),
        )
        legend_labels.append("Reachable")

    legend_handles.append(hold_line)
    legend_labels.append(f"Hold at {hold_at_threshold}")

    ax.legend(legend_handles, legend_labels, frameon=False, loc="lower left")

    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Turn Total (k)")
    ax.set_xlim(0, G)
    ax.set_ylim(0, 50)

    if title is None:
        title = (
            "Cross-section of the roll/hold boundary "
            f"opponent's score = {opponent_score}"
        )

    ax.set_title(title)
    ax.grid(False)

    return ax

def plot_figure5_reachable_states(
    reachable_data: dict[str, np.ndarray],
    ax=None,
    max_points: Optional[int] = 50_000,
    title: str = "States Reachable by an Optimal Pig Player",
):
    """Plot reachable states as a 3D point cloud.

    Args:
        reachable_data:
            Output from figure5_reachable_data(...).
        ax:
            Optional 3D matplotlib axis.
        max_points:
            Optional maximum number of points to plot for speed.
        title:
            Plot title.

    Returns:
        Matplotlib 3D axis.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    points = reachable_data["points"]

    if max_points is not None and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[idx]

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.4)

    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Player 2 Score (j)")
    ax.set_zlabel("Turn Total (k)")
    ax.set_title(title)

    return ax


def plot_figure6_reachable_continue_states(
    reachable_continue_data: dict[str, np.ndarray],
    ax=None,
    max_points: Optional[int] = 50_000,
    title: str = "Reachable States Where Continuing is Optimal",
):
    """Plot reachable states where continuing is optimal.

    Args:
        reachable_continue_data:
            Output from figure6_reachable_continue_data(...).
        ax:
            Optional 3D matplotlib axis.
        max_points:
            Optional maximum number of points to plot for speed.
        title:
            Plot title.

    Returns:
        Matplotlib 3D axis.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    points = reachable_continue_data["points"]

    if max_points is not None and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[idx]

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.4)

    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Player 2 Score (j)")
    ax.set_zlabel("Turn Total (k)")
    ax.set_title(title)

    return ax


def _plot_probability_contour_with_marching_cubes(
    V: np.ndarray,
    level: float,
    ax,
    valid_mask: np.ndarray,
):
    """Plot one probability contour using marching cubes.

    Args:
        V:
            Value table.
        level:
            Contour level.
        ax:
            3D matplotlib axis.
        valid_mask:
            Valid-state mask.

    Returns:
        None.

    Raises:
        ImportError:
            If scikit-image is not installed.
    """

    from skimage import measure

    volume = np.array(V, copy=True)
    volume[~valid_mask] = np.nan

    # marching_cubes cannot handle NaN. Fill invalid entries far below the
    # contour range, then rely on the valid-state geometry to keep the useful
    # surface inside the state space.
    filled = np.nan_to_num(volume, nan=-1.0)

    verts, faces, _, _ = measure.marching_cubes(filled, level=level)

    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        linewidth=0.2,
        alpha=0.35,
    )


def _plot_probability_contour_fallback(
    V: np.ndarray,
    level: float,
    ax,
    valid_mask: np.ndarray,
    tolerance: float = 0.005,
    max_points: int = 20_000,
):
    """Fallback contour plot using near-level point cloud.

    Args:
        V:
            Value table.
        level:
            Probability level.
        ax:
            3D matplotlib axis.
        valid_mask:
            Valid-state mask.
        tolerance:
            Points satisfying |V-level| <= tolerance are shown.
        max_points:
            Maximum number of points plotted.

    Returns:
        None.
    """

    near = np.logical_and(valid_mask, np.abs(V - level) <= tolerance)
    points = np.argwhere(near)

    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[idx]

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.35)


def plot_figure7_probability_contours(
    contour_data: dict[str, object],
    ax=None,
    title: str = "Win Probability Contours for Optimal Play",
):
    """Plot probability contours for optimal play.

    Args:
        contour_data:
            Output from figure7_probability_contour_data(...).
        ax:
            Optional 3D matplotlib axis.
        title:
            Plot title.

    Returns:
        Matplotlib 3D axis.

    Notes:
        The paper shows contours at 3%, 9%, 27%, and 81%. This function tries
        to use marching cubes for surface contours. If scikit-image is not
        installed, it falls back to plotting near-level point clouds.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    V = contour_data["V_filled"]
    levels = contour_data["levels"]
    valid_mask = contour_data["valid_mask"]

    for level in levels:
        try:
            _plot_probability_contour_with_marching_cubes(
                V=V,
                level=float(level),
                ax=ax,
                valid_mask=valid_mask,
            )
        except Exception:
            _plot_probability_contour_fallback(
                V=V,
                level=float(level),
                ax=ax,
                valid_mask=valid_mask,
            )

    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Player 2 Score (j)")
    ax.set_zlabel("Turn Total (k)")
    ax.set_title(title)

    return ax


# ---------------------------------------------------------------------
# 6. Convenience summary helpers
# ---------------------------------------------------------------------

def summarize_solution(spec: dict, V: np.ndarray, policy: np.ndarray, restricted_k: bool) -> dict[str, object]:
    """Summarize a solved value-iteration result.

    Args:
        spec:
            Game specification.
        V:
            Value table.
        policy:
            Policy table.

    Returns:
        Dictionary with useful summary statistics:
            target_score
            n_valid_states
            start_win_probability
            n_continue_states
            n_hold_states
    """

    valid = vi.valid_state_mask(spec, restricted_k = restricted_k)

    return {
        "target_score": vi.target_score(spec),
        "n_valid_states": int(valid.sum()),
        "start_win_probability": float(V[0, 0, 0]),
        "n_continue_states": int(np.logical_and(valid, policy == 1).sum()),
        "n_hold_states": int(np.logical_and(valid, policy == 0).sum()),
    }


def check_pig_start_probability(
    V: np.ndarray,
    expected: float = 0.5306,
    atol: float = 5e-4,
) -> float:
    """Check Pig starting-player win probability against the paper's value.

    Args:
        V:
            Solved Pig value table.
        expected:
            Paper value, approximately 0.5306.
        atol:
            Absolute tolerance.

    Returns:
        Absolute error.

    Raises:
        AssertionError:
            If the error exceeds atol.
    """

    got = float(V[0, 0, 0])
    err = abs(got - expected)

    if err > atol:
        raise AssertionError(
            f"P[0,0,0] = {got}, expected approximately {expected}, error {err}"
        )

    return err
