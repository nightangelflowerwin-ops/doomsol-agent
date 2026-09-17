"""Small, model-independent helpers for Doom state-dependence checks."""

from __future__ import annotations

import math

import numpy as np
import torch
import vizdoom as vzd


ENEMY_NAMES = {
    "MarineChainsawVzd", "Zombieman", "ShotgunGuy", "ChaingunGuy",
    "DoomImp", "Demon", "Spectre", "Cacodemon", "LostSoul", "BaronOfHell",
    "HellKnight", "Revenant", "Mancubus", "Arachnotron", "PainElemental",
    "Archvile", "Cyberdemon", "SpiderMastermind",
}


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(p.dtype).eps
    return (p * ((p + epsilon).log() - (q + epsilon).log())).sum(-1)


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _angle_delta(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def visible_enemy_observations(state, game=None) -> list[dict[str, float | bool | str]]:
    """Return authoritative visible-enemy geometry for teacher labels and audits.

    ViZDoom's labels buffer contains only rendered, unoccluded objects. Doom sight
    tests are symmetric, so a labelled enemy has a clear line segment to the
    player. The policy never receives this structure; it is teacher-only truth.
    """
    enemies = [label for label in (state.labels or []) if label.object_name in ENEMY_NAMES]
    height, width = state.screen_buffer.shape[:2]
    player_x = player_y = None
    if game is not None:
        player_x = float(game.get_game_variable(vzd.GameVariable.POSITION_X))
        player_y = float(game.get_game_variable(vzd.GameVariable.POSITION_Y))
    result = []
    for enemy in enemies:
        centre = enemy.x + enemy.width / 2
        if player_x is None:
            distance = math.sqrt(max(1.0, width * height) /
                                 max(1.0, enemy.width * enemy.height))
            facing_agent = False
        else:
            dx = player_x - float(enemy.object_position_x)
            dy = player_y - float(enemy.object_position_y)
            distance = math.hypot(dx, dy)
            bearing_to_player = math.degrees(math.atan2(dy, dx)) % 360.0
            facing_agent = abs(_angle_delta(bearing_to_player, float(enemy.object_angle))) <= 90.0
        result.append({
            "name": enemy.object_name,
            "x": float(2 * centre / width - 1),
            "area": float(enemy.width * enemy.height / (width * height)),
            "distance": float(distance),
            "mutual_line_of_sight": True,
            "facing_agent": facing_agent,
        })
    return result


def enemy_observation(state, game=None) -> dict[str, float | bool | str] | None:
    enemies = visible_enemy_observations(state, game)
    if not enemies:
        return None
    # Nearest visible enemy is the immediate threat. Screen-centre error then
    # gives the exact turn/fire teacher target.
    return min(enemies, key=lambda enemy: (float(enemy["distance"]), abs(float(enemy["x"]))))
