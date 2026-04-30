"""
Piglet is a simple version of Pig. A player repeatedly flips a coin during a
turn. Heads adds 1 to the current turn total and the turn continues. Tails
ends the turn and scores nothing for that turn. The player may also hold,
adding the turn total to their score and passing the turn.

This file contains only the Piglet game mechanics and a small game specification
used by value_iteration.py. It does not contain the value-iteration solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional
import random


Action = Literal["flip", "hold"]
Policy = Callable[["PigletState", random.Random], Action]


@dataclass(frozen=True)
class PigletState:
    """Immutable state for actually playing a Piglet game.

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
        This is a game-playing state, not the article's value-iteration state.
        In value iteration, a state is represented by a tuple ``(i, j, k)``,
        where ``i`` is the current player's score, ``j`` is the opponent's
        score, and ``k`` is the current turn total.
    """

    scores: tuple[int, int] = (0, 0)
    turn_total: int = 0
    current_player: int = 0
    winner: Optional[int] = None


def make_spec(target_score: int = 2) -> dict:
    """Return the value-iteration specification for Piglet.

    Args:
        target_score:
            Winning score. The article introduces Piglet with goal 10, then
            uses goal 2 as the worked value-iteration example.

    Returns:
        A dictionary consumed by ``value_iteration.py``. The transition model is:

        * bust probability = 1/2 for tails;
        * positive outcome = +1 with probability 1/2 for heads;
        * continue action name = ``"flip"``.

    Formula used later by value iteration:
        ``P_flip(i,j,k) = 0.5 * (1 - P[j,i,0]) + 0.5 * P[i,j,k+1]``.
    """

    if target_score < 2:
        raise ValueError("target_score must be at least 2.")

    return {
        "name": f"piglet_goal_{target_score}",
        "game": "piglet",
        "target_score": target_score,
        "continue_action": "flip",
        "hold_action": "hold",
        "bust_probability": 0.5,
        "gain_outcomes": (1,),
        "gain_probabilities": (0.5,),
    }


def initial_state(first_player: int = 0) -> PigletState:
    """Create an initial Piglet game state.

    Args:
        first_player:
            The player who acts first, either 0 or 1.

    Returns:
        ``PigletState(scores=(0, 0), turn_total=0, current_player=first_player)``.
    """

    if first_player not in (0, 1):
        raise ValueError("first_player must be 0 or 1.")

    return PigletState(
        scores=(0, 0),
        turn_total=0,
        current_player=first_player,
        winner=None,
    )


def legal_actions(state: PigletState) -> tuple[Action, ...]:
    """Return legal actions in the current state.

    Args:
        state:
            Current Piglet game state.

    Returns:
        ``()`` if the game is over; otherwise ``("flip", "hold")``.
    """

    if state.winner is not None:
        return ()

    return ("flip", "hold")


def step(
    state: PigletState,
    action: Action,
    rng: random.Random,
    target_score: int = 2,
) -> PigletState:
    """Advance the Piglet game by one action.

    Args:
        state:
            Current game-playing state.
        action:
            ``"flip"`` or ``"hold"``.
        rng:
            Random-number generator.
        target_score:
            Winning score.

    Returns:
        The next ``PigletState``.

    Rules:
        * ``hold`` banks ``turn_total`` into the current player's score. If the
          new score reaches ``target_score``, that player wins.
        * ``flip`` uses a fair coin. Heads adds 1 to ``turn_total``. Tails loses
          the current ``turn_total`` and passes the turn.
    """

    if state.winner is not None:
        return state

    if action not in legal_actions(state):
        raise ValueError(f"Illegal Piglet action: {action!r}.")

    p = state.current_player
    q = 1 - p
    scores = list(state.scores)

    if action == "hold":
        scores[p] += state.turn_total

        if scores[p] >= target_score:
            return PigletState(
                scores=tuple(scores),
                turn_total=0,
                current_player=p,
                winner=p,
            )

        return PigletState(
            scores=tuple(scores),
            turn_total=0,
            current_player=q,
            winner=None,
        )

    # action == "flip"
    heads = rng.random() < 0.5

    if heads:
        return PigletState(
            scores=state.scores,
            turn_total=state.turn_total + 1,
            current_player=p,
            winner=None,
        )

    return PigletState(
        scores=state.scores,
        turn_total=0,
        current_player=q,
        winner=None,
    )


def play_game(
    policy0: Policy,
    policy1: Policy,
    target_score: int = 2,
    seed: Optional[int] = None,
    first_player: int = 0,
    max_steps: int = 100_000,
) -> PigletState:
    """Play a complete Piglet game between two policies.

    Args:
        policy0:
            Function mapping ``(state, rng)`` to ``"flip"`` or ``"hold"``.
        policy1:
            Same, for player 1.
        target_score:
            Winning score.
        seed:
            Optional RNG seed.
        first_player:
            Starting player, either 0 or 1.
        max_steps:
            Safety cap against non-terminating pathological policies.

    Returns:
        Final ``PigletState`` with ``winner`` set.
    """

    rng = random.Random(seed)
    state = initial_state(first_player=first_player)
    policies = (policy0, policy1)

    for _ in range(max_steps):
        if state.winner is not None:
            return state

        action = policies[state.current_player](state, rng)
        state = step(state, action, rng, target_score=target_score)

    raise RuntimeError("Piglet game exceeded max_steps; check the policies.")


def make_hold_at_policy(threshold: int, target_score: int = 2) -> Policy:
    """Create a simple threshold policy for Piglet.

    Args:
        threshold:
            Hold once ``turn_total >= threshold``.
        target_score:
            Winning score. The policy also holds if holding wins immediately.

    Returns:
        A policy function compatible with ``play_game``.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    def policy(state: PigletState, rng: random.Random) -> Action:
        p = state.current_player

        if state.scores[p] + state.turn_total >= target_score:
            return "hold"

        return "hold" if state.turn_total >= threshold else "flip"

    return policy


def make_always_flip_policy(target_score: int = 2) -> Policy:
    """Create a policy that flips unless holding immediately wins.

    Args:
        target_score:
            Winning score.

    Returns:
        A policy function compatible with ``play_game``.
    """

    def policy(state: PigletState, rng: random.Random) -> Action:
        p = state.current_player

        if state.scores[p] + state.turn_total >= target_score:
            return "hold"

        return "flip"

    return policy