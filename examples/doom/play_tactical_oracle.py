"""Authoritative local combat teacher with simultaneous Doom controls.

The oracle sees ViZDoom object/sector truth and is therefore a data generator,
not a deployable visual policy. Its trace is designed for later distillation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import vizdoom as vzd

from jevlike.provenance import provenance
from diagnostics import ENEMY_NAMES
from environment import TICS_PER_ACTION, TICS_PER_SECOND, action_vector_for_names, make_game


def angle_delta(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def object_distance(a, b) -> float:
    return math.hypot(a.position_x - b.position_x, a.position_y - b.position_y)


def bearing(a, b) -> float:
    return math.degrees(math.atan2(b.position_y - a.position_y,
                                   b.position_x - a.position_x)) % 360.0


class TacticalOracle:
    def __init__(self) -> None:
        self.previous_health = 100.0
        self.last_damage_tic = -10_000
        self.strafe_sign = 1

    def reset(self, health: float) -> None:
        self.previous_health = health
        self.last_damage_tic = -10_000
        self.strafe_sign = 1

    def decide(self, state, game) -> tuple[set[str], dict]:
        objects = list(state.objects or [])
        player = next(item for item in objects if item.name == "DoomPlayer")
        enemies = [item for item in objects if item.name in ENEMY_NAMES]
        visible_ids = {
            label.object_id for label in (state.labels or [])
            if label.object_name in ENEMY_NAMES
        }
        health = float(game.get_game_variable(vzd.GameVariable.HEALTH))
        damaged = health < self.previous_health
        if damaged:
            self.last_damage_tic = state.tic
            self.strafe_sign *= -1
        self.previous_health = health

        if not enemies:
            return {"move forward"}, {
                "mode": "advance", "health": health, "damaged": damaged,
                "enemy_count": 0, "visible_enemies": 0,
            }

        # Visible threats are engaged first. Otherwise rotate toward the nearest
        # known enemy coordinate until Doom's sight line becomes clear.
        target = min(
            enemies,
            key=lambda item: (item.id not in visible_ids, object_distance(player, item)),
        )
        distance = object_distance(player, target)
        target_bearing = bearing(player, target)
        error = angle_delta(target_bearing, float(player.angle))
        line_of_sight = target.id in visible_ids
        incoming_names = [
            item.name for item in objects
            if item.name not in ENEMY_NAMES and item.name not in {"DoomPlayer", "GreenArmor"}
            and "ammo" not in item.name.lower()
            and any(token in item.name.lower() for token in
                    ("ball", "rocket", "plasma", "fire", "tracer", "missile"))
        ]

        actions: set[str] = set()
        if error > 4.0:
            actions.add("turn left")
        elif error < -4.0:
            actions.add("turn right")

        aligned = abs(error) <= 8.0
        under_fire = state.tic - self.last_damage_tic <= 18 or bool(incoming_names)
        if line_of_sight:
            # Maintain lateral motion under fire and never wait in the open.
            phase = (state.tic // 12) % 2
            if under_fire or distance < 420.0:
                direction = self.strafe_sign if phase == 0 else -self.strafe_sign
                actions.add("strafe left" if direction < 0 else "strafe right")
            if distance < 96.0:
                actions.add("move backward")
            elif distance > 320.0 and aligned:
                actions.add("move forward")
            if aligned:
                actions.add("attack")
        else:
            # Search/close only after facing the last authoritative bearing.
            if abs(error) <= 12.0:
                actions.add("move forward")

        if not actions:
            actions.add("turn left" if error >= 0 else "turn right")
        return actions, {
            "mode": "engage" if line_of_sight else "search",
            "target_id": int(target.id), "target_name": target.name,
            "target_distance": round(distance, 3),
            "target_bearing": round(target_bearing, 3),
            "aim_error_degrees": round(error, 3),
            "line_of_sight": line_of_sight,
            "enemy_can_see_agent": line_of_sight,
            "incoming_projectiles": incoming_names,
            "under_fire": under_fire, "health": health, "damaged": damaged,
            "enemy_count": len(enemies), "visible_enemies": len(visible_ids),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--log-every", type=float, default=10.0)
    args = parser.parse_args()

    game = make_game(
        args.seed, "deadly_corridor", "640x480", include_use=True,
        window_visible=args.visible, teacher_truth=True,
    )
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace_file = args.trace.open("w", encoding="utf-8")
    else:
        trace_file = None

    oracle = TacticalOracle()
    limit = round(args.seconds * TICS_PER_SECOND / TICS_PER_ACTION)
    started = time.perf_counter()
    next_log = started
    total_steps = video_frames = video_tics = total_kills = 0
    game.new_episode()
    episodes = 1
    oracle.reset(float(game.get_game_variable(vzd.GameVariable.HEALTH)))
    actions_seen: Counter[str] = Counter()
    episode_reward = 0.0
    try:
        while total_steps < limit:
            if game.is_episode_finished() or game.get_state() is None:
                if episodes:
                    total_kills += int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                game.new_episode()
                episodes += 1
                episode_reward = 0.0
                oracle.reset(float(game.get_game_variable(vzd.GameVariable.HEALTH)))
            state = game.get_state()
            if state is None:
                continue
            frame = np.ascontiguousarray(state.screen_buffer)
            actions, telemetry = oracle.decide(state, game)
            for action in actions:
                actions_seen[action] += 1
            reward = float(game.make_action(
                action_vector_for_names(actions, game), TICS_PER_ACTION
            ))
            episode_reward += reward
            row = {
                **provenance(),
                "step": total_steps, "tic": int(state.tic), "episode": episodes,
                "actions": sorted(actions), "reward": reward,
                "episode_reward": episode_reward,
                "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                **telemetry,
            }
            if trace_file:
                trace_file.write(json.dumps(row) + "\n")
            now = time.perf_counter()
            if now >= next_log:
                print(json.dumps(row), flush=True)
                next_log = now + args.log_every
            video_tics += TICS_PER_ACTION
            target_frames = round(video_tics * 30 / TICS_PER_SECOND)
            if writer:
                while video_frames < target_frames:
                    writer.append_data(frame)
                    video_frames += 1
            total_steps += 1
            elapsed = time.perf_counter() - now
            time.sleep(max(0.0, TICS_PER_ACTION / TICS_PER_SECOND - elapsed))
    finally:
        if not game.is_episode_finished():
            total_kills += int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        if trace_file:
            trace_file.close()
        if writer:
            writer.close()
        game.close()
    print(json.dumps({
        **provenance(),
        "seconds": args.seconds, "episodes": episodes, "steps": total_steps,
        "total_kills": total_kills, "kills_per_episode": total_kills / max(1, episodes),
        "action_counts": dict(actions_seen), "video": str(args.output) if args.output else None,
        "trace": str(args.trace) if args.trace else None,
    }), flush=True)


if __name__ == "__main__":
    main()
