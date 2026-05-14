"""
One-die Pig game definition aligned with Neller and Presser (2004).

Rules:
    The target score is 100. On each turn, a player repeatedly rolls one die.
    Rolling 1 loses the current turn total and passes the turn. Rolling 2--6
    adds that number to the current turn total. Holding banks the turn total
    and passes play, unless the banked score reaches the target score and wins
    the game.

This file contains only game mechanics and the value-iteration specification.
The solver is implemented in value_iteration.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional
import random


Action = Literal["roll", "hold"]
Policy = Callable[["PigState", random.Random], Action]


@dataclass(frozen=True)
class PigState:
    """Immutable state for actually playing a one-die Pig game.

    Attributes:
        scores:
            Pair ``(score_player_0, score_player_1)``.
        turn_total:
            Current unbanked turn total.
        current_player:
            Index of the player to act, either 0 or 1.
        winner:
            ``None`` if the game is not over; otherwise 0 or 1.

    Important:
        This is the game-playing state. The value-iteration state used in the
        article is ``(i, j, k)``, where ``i`` is the current player's score,
        ``j`` is the opponent's score, and ``k`` is the current turn total.
    """

    scores: tuple[int, int] = (0, 0)
    turn_total: int = 0
    current_player: int = 0
    winner: Optional[int] = None


def make_spec(target_score: int = 100) -> dict:
    """Return the value-iteration specification for one-die Pig.

    Args:
        target_score:
            Winning score. The article's main Pig game uses 100.

    Returns:
        A dictionary consumed by ``value_iteration.py``. The transition model is:

        * bust probability = 1/6 for rolling 1;
        * positive outcomes = 2, 3, 4, 5, 6, each with probability 1/6;
        * continue action name = ``"roll"``.

    Formula used later by value iteration:
        ``P_roll(i,j,k) =
            (1/6) * ((1 - P[j,i,0])
                     + P[i,j,k+2] + P[i,j,k+3] + P[i,j,k+4]
                     + P[i,j,k+5] + P[i,j,k+6])``.
    """

    if target_score < 2:
        raise ValueError("target_score must be at least 2.")

    return {
        "name": f"pig_goal_{target_score}",
        "game": "pig",
        "target_score": target_score,
        "continue_action": "roll",
        "hold_action": "hold",
        "bust_probability": 1.0 / 6.0,
        "gain_outcomes": (2, 3, 4, 5, 6),
        "gain_probabilities": (1.0 / 6.0,) * 5,
    }


def initial_state(first_player: int = 0) -> PigState:
    """Create an initial Pig game state.

    Args:
        first_player:
            The player who acts first, either 0 or 1.

    Returns:
        ``PigState(scores=(0, 0), turn_total=0, current_player=first_player)``.
    """

    if first_player not in (0, 1):
        raise ValueError("first_player must be 0 or 1.")

    return PigState(
        scores=(0, 0),
        turn_total=0,
        current_player=first_player,
        winner=None,
    )


def legal_actions(state: PigState) -> tuple[Action, ...]:
    """Return legal actions in the current state.

    Args:
        state:
            Current Pig game state.

    Returns:
        ``()`` if the game is over; otherwise ``("roll", "hold")``.
    """

    if state.winner is not None:
        return ()

    return ("roll", "hold")


def step(
    state: PigState,
    action: Action,
    rng: random.Random,
    target_score: int = 100,
) -> PigState:
    """Advance the Pig game by one action.

    Args:
        state:
            Current game-playing state.
        action:
            ``"roll"`` or ``"hold"``.
        rng:
            Random-number generator.
        target_score:
            Winning score.

    Returns:
        The next ``PigState``.

    Rules:
        * ``hold`` banks ``turn_total`` into the current player's score. If the
          new score reaches ``target_score``, that player wins.
        * ``roll`` draws one fair die. A 1 loses the current turn total and
          passes the turn. A 2--6 is added to ``turn_total``.
    """

    if state.winner is not None:
        return state

    if action not in legal_actions(state):
        raise ValueError(f"Illegal Pig action: {action!r}.")

    p = state.current_player
    q = 1 - p
    scores = list(state.scores)

    if action == "hold":
        scores[p] += state.turn_total

        if scores[p] >= target_score:
            return PigState(
                scores=tuple(scores),
                turn_total=0,
                current_player=p,
                winner=p,
            )

        return PigState(
            scores=tuple(scores),
            turn_total=0,
            current_player=q,
            winner=None,
        )

    # action == "roll"
    die = rng.randint(1, 6)

    if die == 1:
        return PigState(
            scores=state.scores,
            turn_total=0,
            current_player=q,
            winner=None,
        )

    return PigState(
        scores=state.scores,
        turn_total=state.turn_total + die,
        current_player=p,
        winner=None,
    )


def play_game(
    policy0: Policy,
    policy1: Policy,
    target_score: int = 100,
    seed: Optional[int] = None,
    first_player: int = 0,
    max_steps: int = 1_000_000,
) -> PigState:
    """Play a complete Pig game between two policies.

    Args:
        policy0:
            Function mapping ``(state, rng)`` to ``"roll"`` or ``"hold"``.
        policy1:
            Same, for player 1.
        target_score:
            Winning score.
        seed:
            Optional RNG seed.
        first_player:
            Starting player.
        max_steps:
            Safety cap against non-terminating pathological policies.

    Returns:
        Final ``PigState`` with ``winner`` set.
    """

    rng = random.Random(seed)
    state = initial_state(first_player=first_player)
    policies = (policy0, policy1)

    for _ in range(max_steps):
        if state.winner is not None:
            return state

        action = policies[state.current_player](state, rng)
        state = step(state, action, rng, target_score=target_score)

    raise RuntimeError("Pig game exceeded max_steps; check the policies.")


def make_hold_at_policy(threshold: int = 20, target_score: int = 100) -> Policy:
    """Create a hold-at-threshold Pig policy.

    Args:
        threshold:
            Hold once ``turn_total >= threshold``. The article discusses the
            common baseline ``threshold = 20``.
        target_score:
            Winning score. The policy also holds if holding wins immediately.

    Returns:
        A policy function compatible with ``play_game``.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    def policy(state: PigState, rng: random.Random) -> Action:
        p = state.current_player

        if state.scores[p] + state.turn_total >= target_score:
            return "hold"

        return "hold" if state.turn_total >= threshold else "roll"

    return policy


def hold_at_20_policy(state: PigState, rng: random.Random) -> Action:
    """Article baseline policy: hold at turn total 20.

    Args:
        state:
            Current Pig state.
        rng:
            Random-number generator, unused.

    Returns:
        ``"hold"`` if the current player can win immediately or if
        ``turn_total >= 20``; otherwise ``"roll"``.

    Note:
        This function assumes the article's target score 100.
    """

    p = state.current_player

    if state.scores[p] + state.turn_total >= 100:
        return "hold"

    return "hold" if state.turn_total >= 20 else "roll"