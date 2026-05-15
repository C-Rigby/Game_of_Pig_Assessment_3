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

from . import value_iteration_fun as vi

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



def _figure3_transition_surfaces(
    policy: np.ndarray,
) -> tuple[list[dict[str, np.ndarray | int]], np.ndarray]:
    """Return k-direction roll/hold transition sheets for Figure 3.

    Figure 3 in the paper is best approximated by the action changes that
    occur as the turn total ``k`` increases for each fixed score pair
    ``(i,j)``. This preserves overhangs by keeping the first transition, second
    transition, and so on as separate sheets, while avoiding the unrelated
    i/j-direction voxel walls that made the figure look blocky.
    """

    if policy.ndim != 3:
        raise ValueError("policy must be a 3D array")

    G_i, G_j, G_k = (int(x) for x in policy.shape)
    max_transitions = 0
    layers: dict[int, list[tuple[int, int, float, int, int]]] = {}
    rows: list[tuple[float, float, float, float, float, float]] = []

    for i in range(G_i):
        terminal_k = G_i - i if G_i == G_k else G_k

        for j in range(G_j):
            valid_k = np.where(policy[i, j, :] >= 0)[0]

            if valid_k.size < 2:
                continue

            transition_order = 0
            previous_action = int(policy[i, j, int(valid_k[0])])

            for k_value in valid_k[1:]:
                k = int(k_value)
                action = int(policy[i, j, k])

                if action != previous_action:
                    transition_order += 1

                    # The paper draws the immediate-win limit i+k=G as its own
                    # dark plane, so do not also plot policy transitions inside
                    # or on that terminal region.
                    if 0 <= k < terminal_k:
                        max_transitions = max(max_transitions, transition_order)
                        layers.setdefault(transition_order, []).append(
                            (i, j, float(k), previous_action, action)
                        )
                        rows.append(
                            (
                                float(i),
                                float(j),
                                float(k),
                                float(previous_action),
                                float(action),
                                float(transition_order),
                            )
                        )

                previous_action = action

    surfaces: list[dict[str, np.ndarray | int]] = []

    for transition_order in range(1, max_transitions + 1):
        K = np.full((G_i, G_j), np.nan, dtype=float)
        points: list[tuple[float, float, float]] = []

        for i, j, k, _from_action, _to_action in layers.get(transition_order, []):
            K[i, j] = k
            points.append((float(i), float(j), float(k)))

        surfaces.append(
            {
                "transition_order": transition_order,
                "K": K,
                "points": np.asarray(points, dtype=float),
            }
        )

    if rows:
        transition_points = np.asarray(rows, dtype=float)
    else:
        transition_points = np.empty((0, 6), dtype=float)

    return surfaces, transition_points



def _figure3_surface_polygons(
    I: np.ndarray,
    J: np.ndarray,
    K: np.ndarray,
    max_surface_jump: Optional[float] = None,
) -> list[np.ndarray]:
    """Convert one transition sheet into edge-free quadrilateral polygons."""

    polygons: list[np.ndarray] = []
    rows, cols = K.shape

    for i in range(rows - 1):
        for j in range(cols - 1):
            z = np.asarray(
                [K[i, j], K[i + 1, j], K[i + 1, j + 1], K[i, j + 1]],
                dtype=float,
            )

            if not np.isfinite(z).all():
                continue

            if (
                max_surface_jump is not None
                and float(np.nanmax(z) - np.nanmin(z)) > max_surface_jump
            ):
                continue

            polygons.append(
                np.asarray(
                    [
                        [I[i, j], J[i, j], z[0]],
                        [I[i + 1, j], J[i + 1, j], z[1]],
                        [I[i + 1, j + 1], J[i + 1, j + 1], z[2]],
                        [I[i, j + 1], J[i, j + 1], z[3]],
                    ],
                    dtype=float,
                )
            )

    return polygons



def _figure3_terminal_polygon(policy: np.ndarray) -> list[np.ndarray]:
    """Return the unrestricted i + k = G terminal plane used in Figure 3."""

    if policy.shape[2] != policy.shape[0]:
        return []

    i_max = float(policy.shape[0])
    j_max = float(policy.shape[1])
    k_max = float(policy.shape[2])

    return [
        np.asarray(
            [
                [0.0, 0.0, k_max],
                [0.0, j_max, k_max],
                [i_max, j_max, 0.0],
                [i_max, 0.0, 0.0],
            ],
            dtype=float,
        )
    ]



def _figure3_interface_face_polygon(
    i: int,
    j: int,
    k: int,
    axis: int,
) -> np.ndarray:
    x0, x1 = float(i), float(i + 1)
    y0, y1 = float(j), float(j + 1)
    z0, z1 = float(k), float(k + 1)

    if axis == 0:
        x = x1
        return np.asarray(
            [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]],
            dtype=float,
        )

    if axis == 1:
        y = y1
        return np.asarray(
            [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            dtype=float,
        )

    z = z1
    return np.asarray(
        [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
        dtype=float,
    )



def _figure3_policy_interface_polygons(
    policy: np.ndarray,
    max_faces: Optional[int] = None,
    include_ij_walls: bool = False,
    include_k_walls: bool = True,
    exclude_terminal_region: bool = True,
) -> list[np.ndarray]:
    if policy.ndim != 3:
        raise ValueError("policy must be a 3D array")

    G_i, G_j, G_k = (int(x) for x in policy.shape)
    polygons: list[np.ndarray] = []

    axes: list[int] = []
    if include_ij_walls:
        axes.extend([0, 1])
    if include_k_walls:
        axes.append(2)

    for axis in axes:
        upper = [G_i, G_j, G_k]
        upper[axis] -= 1

        for i in range(upper[0]):
            for j in range(upper[1]):
                for k in range(upper[2]):
                    current = int(policy[i, j, k])

                    if current < 0:
                        continue

                    ni, nj, nk = i, j, k
                    if axis == 0:
                        ni += 1
                    elif axis == 1:
                        nj += 1
                    else:
                        nk += 1

                    neighbour = int(policy[ni, nj, nk])

                    if neighbour < 0:
                        continue

                    if current == neighbour:
                        continue

                    if exclude_terminal_region and G_i == G_k:
                        if i + k >= G_i or ni + nk >= G_i:
                            continue

                    polygons.append(
                        _figure3_interface_face_polygon(
                            i,
                            j,
                            k,
                            axis=axis,
                        )
                    )

    if max_faces is not None and len(polygons) > max_faces:
        idx = np.linspace(0, len(polygons) - 1, max_faces).astype(int)
        polygons = [polygons[int(x)] for x in idx]

    return polygons



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

        if not vi.is_valid_state(spec, score0, score1, k, restricted_k=restricted_k):
            return []

        action_code = int(policy[score0, score1, k])
        actions = ["continue"] if action_code == 1 else ["hold"]

    else:
        if score1 + k >= G:
            return []

        if opponent_mode == "any":
            actions = ["continue", "hold"]
        elif opponent_mode == "same_policy":
            if not vi.is_valid_state(spec, score1, score0, k, restricted_k=restricted_k):
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

        if player == 0 and vi.is_valid_state(spec, score0, score1, k, restricted_k=restricted_k):
            reachable[score0, score1, k] = True

        for nxt in _expand_actual_state(spec, policy, actual, opponent_mode, restricted_k=restricted_k):
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



def _voxel_face_polygon(
    i: int,
    j: int,
    k: int,
    axis: int,
    positive_side: bool,
) -> np.ndarray:
    """Return one exposed voxel face as a 3D quadrilateral."""

    x0, x1 = float(i), float(i + 1)
    y0, y1 = float(j), float(j + 1)
    z0, z1 = float(k), float(k + 1)

    if axis == 0:
        x = x1 if positive_side else x0
        return np.asarray(
            [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]],
            dtype=float,
        )

    if axis == 1:
        y = y1 if positive_side else y0
        return np.asarray(
            [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            dtype=float,
        )

    z = z1 if positive_side else z0
    return np.asarray(
        [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
        dtype=float,
    )



def _mask_surface_faces(
    mask: np.ndarray,
    max_faces: Optional[int] = None,
) -> list[np.ndarray]:
    """Return exposed faces for a 3D Boolean mask."""

    mask_array = np.asarray(mask, dtype=bool)

    if mask_array.ndim != 3:
        raise ValueError("mask must be a 3D array")

    padded = np.pad(mask_array, 1, mode="constant", constant_values=False)
    interior = tuple(slice(1, -1) for _ in range(3))
    current = padded[interior]
    faces: list[np.ndarray] = []

    for axis in range(3):
        for positive_side in (False, True):
            neighbor_slice = [slice(1, -1)] * 3

            if positive_side:
                neighbor_slice[axis] = slice(2, None)
            else:
                neighbor_slice[axis] = slice(None, -2)

            exposed = np.logical_and(
                current,
                np.logical_not(padded[tuple(neighbor_slice)]),
            )

            for i, j, k in np.argwhere(exposed):
                faces.append(
                    _voxel_face_polygon(
                        int(i),
                        int(j),
                        int(k),
                        axis=axis,
                        positive_side=positive_side,
                    )
                )

    if max_faces is not None and len(faces) > max_faces:
        idx = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces = [faces[int(x)] for x in idx]

    return faces


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



def figure3_boundary_data(
    policy: np.ndarray,
    include_terminal_boundary: Optional[bool] = None,
) -> dict[str, object]:
    """Extract Figure 3 roll/hold boundary data.

    Args:
        policy:
            Policy table.
        include_terminal_boundary:
            Whether to include the i + k = G plane visible in the paper's
            unrestricted Figure 3. If omitted, it is included when the policy
            has no invalid entries and the score and turn-total dimensions
            match.

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
            transition_surfaces:
                Separate k-direction action-change sheets grouped by
                transition order.
            transition_points:
                Rows ``(i, j, k, from_action, to_action, transition_order)``.
    """

    B = hold_boundary(policy)
    G = B.shape[0]
    I, J = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    transition_surfaces, transition_points = _figure3_transition_surfaces(policy)

    if include_terminal_boundary is None:
        include_terminal_boundary = (
            policy.shape[0] == policy.shape[2]
            and not np.any(policy < 0)
        )

    if include_terminal_boundary:
        terminal_polygons = _figure3_terminal_polygon(policy)
    else:
        terminal_polygons = []

    return {
        "I": I,
        "J": J,
        "K": B,
        "points": boundary_points(policy),
        "transition_surfaces": transition_surfaces,
        "transition_points": transition_points,
        "terminal_polygons": terminal_polygons,
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
        restricted_k=restricted_k,
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
        restricted_k=restricted_k,
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
        "valid_mask": vi.valid_state_mask(spec, restricted_k=restricted_k),
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



def _add_surface_collection(
    ax,
    polygons: list[np.ndarray],
    color: str,
    alpha: float,
):
    """Add shaded quadrilateral polygons to a 3D Matplotlib axis."""

    if not polygons:
        return

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import LightSource

    try:
        collection = Poly3DCollection(
            polygons,
            facecolors=color,
            linewidths=0,
            alpha=alpha,
            antialiaseds=False,
            shade=True,
            lightsource=LightSource(azdeg=315, altdeg=35),
            zsort="average",
        )
    except TypeError:
        collection = Poly3DCollection(
            polygons,
            facecolors=color,
            linewidths=0,
            alpha=alpha,
            antialiaseds=False,
            zsort="average",
        )

    collection.set_edgecolor("none")
    ax.add_collection3d(collection)



def _add_flat_terminal_plane(
    ax,
    polygons: list[np.ndarray],
    color: str,
    alpha: float,
):
    """Add a flat transparent terminal plane without lighting or shading."""

    if not polygons:
        return

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgba

    rgba = to_rgba(color, alpha)
    collection = Poly3DCollection(
        polygons,
        facecolors=[rgba] * len(polygons),
        edgecolors="none",
        linewidths=0,
        antialiaseds=False,
        zsort="min",
    )
    collection.set_edgecolor("none")
    ax.add_collection3d(collection)



def _finish_3d_state_plot(
    ax,
    shape: tuple[int, int, int],
    title: Optional[str],
):
    """Apply common axes settings for 3D state-space plots."""

    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Player 2 Score (j)")
    ax.set_zlabel("Turn Total (k)")
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(shape)

    if title:
        ax.set_title(title)

    return ax



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
    boundary_data: dict[str, object],
    ax=None,
    title: Optional[str] = None,
    boundary_color: str = "#74879B",
    terminal_color: str = "#DCE4EC",
    boundary_alpha: float = 0.88,
    terminal_alpha: float = 0.10,
    max_faces: Optional[int] = None,
    max_surface_jump: Optional[float] = None,
):
    """Plot a paper-style 3D roll/hold boundary surface.

    Args:
        boundary_data:
            Output from figure3_boundary_data(...).
        ax:
            Optional 3D matplotlib axis.
        title:
            Plot title.
        boundary_color:
            Colour for the roll/hold interface.
        terminal_color:
            Colour for the i + k = G terminal plane.
        boundary_alpha:
            Opacity for the roll/hold interface.
        terminal_alpha:
            Opacity for the terminal surface.
        max_faces:
            Optional cap on boundary polygons for quick previews. By default
            all polygons are plotted.
        max_surface_jump:
            Maximum local k-range allowed inside one quadrilateral. This keeps
            separate transition sheets from being bridged by accidental long
            polygons.

    Returns:
        Matplotlib 3D axis.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    I = np.asarray(boundary_data["I"], dtype=float)
    J = np.asarray(boundary_data["J"], dtype=float)
    K = np.asarray(boundary_data["K"], dtype=float)
    transition_points = np.asarray(
        boundary_data.get("transition_points", np.empty((0, 6))),
        dtype=float,
    )
    terminal_polygons = list(boundary_data.get("terminal_polygons", []))
    boundary_points_array = np.asarray(
        boundary_data.get("points", np.empty((0, 3))),
        dtype=float,
    )

    max_i = float(I.shape[0])
    max_j = float(J.shape[1])
    max_k = float(K.shape[0])

    _add_flat_terminal_plane(
        ax,
        terminal_polygons,
        color=terminal_color,
        alpha=terminal_alpha,
    )

    finite = np.isfinite(K)
    if finite.any():
        surface_K = np.ma.masked_invalid(K)
        ax.plot_surface(
            I,
            J,
            surface_K,
            color=boundary_color,
            edgecolor="none",
            linewidth=0,
            alpha=min(float(boundary_alpha), 0.42),
            antialiased=False,
            shade=False,
            rstride=1,
            cstride=1,
        )

    if transition_points.size:
        points = transition_points[:, :3]

        if max_faces is not None and points.shape[0] > max_faces:
            idx = np.linspace(0, points.shape[0] - 1, max_faces).astype(int)
            points = points[idx]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1.6,
            marker="s",
            color="#566577",
            alpha=0.72,
            depthshade=False,
            linewidths=0,
        )

    if boundary_points_array.size:
        if max_faces is not None and boundary_points_array.shape[0] > max_faces:
            idx = np.linspace(0, boundary_points_array.shape[0] - 1, max_faces).astype(int)
            boundary_points_array = boundary_points_array[idx]

        ax.scatter(
            boundary_points_array[:, 0],
            boundary_points_array[:, 1],
            boundary_points_array[:, 2],
            s=2.2,
            marker="s",
            color="#4E5B6A",
            alpha=0.92,
            depthshade=False,
            linewidths=0,
        )

    ax.set_xlabel("Player 1 Score (i)")
    ax.set_ylabel("Player 2 Score (j)")
    ax.set_zlabel("Turn Total (k)")
    ax.set_xlim(0, max_i)
    ax.set_ylim(0, max_j)
    ax.set_zlim(0, max_k)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((max_i, max_j, max_k))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#c8c8c8")
        axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.8

    ax.grid(True)

    if title:
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
    surface: bool = True,
    title: str = "States Reachable by an Optimal Pig Player",
):
    """Plot reachable states as a 3D surface, with scatter fallback.

    Args:
        reachable_data:
            Output from figure5_reachable_data(...).
        ax:
            Optional 3D matplotlib axis.
        max_points:
            Optional maximum number of faces or points to plot for speed.
        surface:
            Whether to draw the exposed surface of the reachable mask.
        title:
            Plot title.

    Returns:
        Matplotlib 3D axis.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    reachable = reachable_data.get("reachable")

    if surface and reachable is not None:
        reachable_array = np.asarray(reachable, dtype=bool)
        faces = _mask_surface_faces(reachable_array, max_faces=max_points)
        _add_surface_collection(ax, faces, color="#8a8a8a", alpha=0.55)

        return _finish_3d_state_plot(ax, reachable_array.shape, title)

    points = reachable_data["points"]

    if max_points is not None and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[idx]

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.4)

    shape = tuple(int(x) for x in np.asarray(reachable).shape) if reachable is not None else (100, 100, 100)
    return _finish_3d_state_plot(ax, shape, title)



def plot_figure6_reachable_continue_states(
    reachable_continue_data: dict[str, np.ndarray],
    ax=None,
    max_points: Optional[int] = 50_000,
    surface: bool = True,
    title: str = "Reachable States Where Continuing is Optimal",
):
    """Plot reachable-roll states as a 3D surface, with scatter fallback.

    Args:
        reachable_continue_data:
            Output from figure6_reachable_continue_data(...).
        ax:
            Optional 3D matplotlib axis.
        max_points:
            Optional maximum number of faces or points to plot for speed.
        surface:
            Whether to draw the exposed surface of the reachable-roll mask.
        title:
            Plot title.

    Returns:
        Matplotlib 3D axis.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    reachable_continue = reachable_continue_data.get("reachable_continue")

    if surface and reachable_continue is not None:
        reachable_continue_array = np.asarray(reachable_continue, dtype=bool)
        faces = _mask_surface_faces(reachable_continue_array, max_faces=max_points)
        _add_surface_collection(ax, faces, color="#5f5f5f", alpha=0.6)

        return _finish_3d_state_plot(ax, reachable_continue_array.shape, title)

    points = reachable_continue_data["points"]

    if max_points is not None and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[idx]

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.4)

    reachable = reachable_continue_data.get("reachable")
    shape_source = reachable_continue if reachable_continue is not None else reachable
    shape = tuple(int(x) for x in np.asarray(shape_source).shape) if shape_source is not None else (100, 100, 100)
    return _finish_3d_state_plot(ax, shape, title)



def _plot_probability_contour_with_marching_cubes(
    V: np.ndarray,
    level: float,
    ax,
    valid_mask: np.ndarray,
    color: str = "#7a7a7a",
    alpha: float = 0.28,
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
        color=color,
        alpha=alpha,
    )



def _level_surface_face_polygon(
    i: int,
    j: int,
    k: int,
    axis: int,
    coordinate: float,
) -> np.ndarray:
    """Return an approximate level-crossing face for a 3D scalar field."""

    x0, x1 = float(i), float(i + 1)
    y0, y1 = float(j), float(j + 1)
    z0, z1 = float(k), float(k + 1)

    if axis == 0:
        x = float(coordinate)
        return np.asarray(
            [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]],
            dtype=float,
        )

    if axis == 1:
        y = float(coordinate)
        return np.asarray(
            [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            dtype=float,
        )

    z = float(coordinate)
    return np.asarray(
        [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
        dtype=float,
    )



def _level_surface_faces(
    V: np.ndarray,
    level: float,
    valid_mask: np.ndarray,
    max_faces: Optional[int] = 20_000,
) -> list[np.ndarray]:
    """Return a lightweight surface approximation for one probability level."""

    values = np.asarray(V, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    faces: list[np.ndarray] = []

    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(0, values.shape[axis] - 1)
        second_slice[axis] = slice(1, values.shape[axis])

        v0 = values[tuple(first_slice)]
        v1 = values[tuple(second_slice)]
        valid_pair = np.logical_and(valid[tuple(first_slice)], valid[tuple(second_slice)])
        delta = v1 - v0
        crosses = np.logical_and(valid_pair, delta != 0)
        crosses = np.logical_and(crosses, (v0 - level) * (v1 - level) <= 0)

        for i, j, k in np.argwhere(crosses):
            left = float(v0[i, j, k])
            right = float(v1[i, j, k])
            t = (float(level) - left) / (right - left)
            coordinate = [float(i), float(j), float(k)][axis] + t
            faces.append(
                _level_surface_face_polygon(
                    int(i),
                    int(j),
                    int(k),
                    axis=axis,
                    coordinate=coordinate,
                )
            )

    if max_faces is not None and len(faces) > max_faces:
        idx = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces = [faces[int(x)] for x in idx]

    return faces



def _plot_probability_contour_fallback(
    V: np.ndarray,
    level: float,
    ax,
    valid_mask: np.ndarray,
    color: str = "#7a7a7a",
    alpha: float = 0.28,
    max_faces: Optional[int] = 20_000,
):
    """Fallback contour plot using approximate level-crossing surfaces.

    Args:
        V:
            Value table.
        level:
            Probability level.
        ax:
            3D matplotlib axis.
        valid_mask:
            Valid-state mask.
        color:
            Surface color.
        alpha:
            Surface opacity.
        max_faces:
            Maximum number of faces plotted.

    Returns:
        None.
    """

    faces = _level_surface_faces(
        V=V,
        level=level,
        valid_mask=valid_mask,
        max_faces=max_faces,
    )

    _add_surface_collection(ax, faces, color=color, alpha=alpha)



def plot_figure7_probability_contours(
    contour_data: dict[str, object],
    ax=None,
    title: str = "Win Probability Contours for Optimal Play",
    max_faces_per_level: Optional[int] = 20_000,
):
    """Plot probability contours for optimal play.

    Args:
        contour_data:
            Output from figure7_probability_contour_data(...).
        ax:
            Optional 3D matplotlib axis.
        title:
            Plot title.
        max_faces_per_level:
            Maximum number of fallback surface faces per probability level.

    Returns:
        Matplotlib 3D axis.

    Notes:
        The paper shows contours at 3%, 9%, 27%, and 81%. This function tries
        to use marching cubes for surface contours. If scikit-image is not
        installed, it uses a lightweight surface approximation rather than a
        point cloud.
    """

    plt = _get_pyplot()

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    V = contour_data["V_filled"]
    levels = contour_data["levels"]
    valid_mask = contour_data["valid_mask"]
    colors = ("#bdbdbd", "#969696", "#737373", "#525252")

    for idx, level in enumerate(levels):
        color = colors[idx % len(colors)]

        try:
            _plot_probability_contour_with_marching_cubes(
                V=V,
                level=float(level),
                ax=ax,
                valid_mask=valid_mask,
                color=color,
                alpha=0.30,
            )
        except Exception:
            _plot_probability_contour_fallback(
                V=V,
                level=float(level),
                ax=ax,
                valid_mask=valid_mask,
                color=color,
                alpha=0.30,
                max_faces=max_faces_per_level,
            )

    return _finish_3d_state_plot(ax, tuple(int(x) for x in np.asarray(V).shape), title)


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

    valid = vi.valid_state_mask(spec, restricted_k=restricted_k)

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
