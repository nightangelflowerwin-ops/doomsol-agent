"""Bounded FreeDoom campaign attempt using combat truth and WAD objectives."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import vizdoom as vzd

from jevlike.provenance import provenance
from environment import TICS_PER_ACTION, TICS_PER_SECOND, action_vector_for_names
from play_tactical_oracle import TacticalOracle, angle_delta, bearing
from wad_navigation import DOOR_SPECIALS, Portal, WadMap


BUTTONS = (
    vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK, vzd.Button.USE,
)
KEY_NAMES = {"BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull"}
HEALTH_NAMES = {"Medikit", "Stimpack", "HealthBonus", "Berserk"}
ARMOR_NAMES = {"GreenArmor", "BlueArmor", "ArmorBonus"}


def make_campaign_game(wad: Path, map_name: str, seed: int, visible: bool,
                       timeout_seconds: float, skill: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.set_doom_game_path(str(wad))
    game.set_doom_map(map_name)
    game.set_doom_skill(skill)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(visible)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    for button in BUTTONS:
        game.add_available_button(button)
    for variable in (
        vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.ITEMCOUNT, vzd.GameVariable.SECRETCOUNT,
        vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE,
    ):
        game.add_available_game_variable(variable)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_episode_timeout(round(timeout_seconds * TICS_PER_SECOND))
    game.set_seed(seed)
    game.init()
    return game


class CampaignNavigator:
    def __init__(self, map_truth: WadMap) -> None:
        self.map = map_truth
        self.objective: tuple[float, float, str] | None = None
        self.route: deque[Portal] = deque()
        self.last_positions: deque[tuple[float, float]] = deque(maxlen=24)
        self.distance_history: deque[float] = deque(maxlen=24)
        self.tracked_waypoint: tuple[int, float, float] | None = None
        self.recovery_sign = 1
        self.recovery_direction = 1
        self.recovery_steps = 0

    def _choose_objective(self, state, player, health: float,
                          armor: float) -> tuple[float, float, str]:
        if health < 55:
            supplies = [item for item in (state.objects or []) if item.name in HEALTH_NAMES]
            if supplies:
                item = min(supplies, key=lambda value: math.hypot(
                    value.position_x - player.position_x, value.position_y - player.position_y
                ))
                return float(item.position_x), float(item.position_y), item.name
        if armor < 20:
            supplies = [item for item in (state.objects or []) if item.name in ARMOR_NAMES]
            if supplies:
                item = min(supplies, key=lambda value: math.hypot(
                    value.position_x - player.position_x, value.position_y - player.position_y
                ))
                if math.hypot(item.position_x - player.position_x,
                              item.position_y - player.position_y) < 700:
                    return float(item.position_x), float(item.position_y), item.name
        keys = [item for item in (state.objects or []) if item.name in KEY_NAMES]
        if keys:
            key = min(keys, key=lambda item: math.hypot(
                item.position_x - player.position_x, item.position_y - player.position_y
            ))
            return float(key.position_x), float(key.position_y), key.name
        x, y, _ = self.map.exit_points[0]
        return x, y, "exit"

    def decide(self, state, health: float, armor: float) -> tuple[set[str], dict]:
        player = next(item for item in state.objects if item.name == "DoomPlayer")
        current_objective = self._choose_objective(state, player, health, armor)
        objective_changed = self.objective != current_objective
        self.objective = current_objective
        self.last_positions.append((player.position_x, player.position_y))

        if objective_changed or not self.route:
            self.route = deque(self.map.route(
                (player.position_x, player.position_y), current_objective[:2]
            ))
        while self.route and math.hypot(
            self.route[0].x - player.position_x,
            self.route[0].y - player.position_y,
        ) < 34.0:
            self.route.popleft()
        if not self.route:
            waypoint = Portal(-1, current_objective[0], current_objective[1], 0)
        else:
            waypoint = self.route[0]

        desired = bearing(player, type("Waypoint", (), {
            "position_x": waypoint.x, "position_y": waypoint.y
        })())
        error = angle_delta(desired, float(player.angle))
        distance = math.hypot(waypoint.x - player.position_x, waypoint.y - player.position_y)
        waypoint_key = (waypoint.target_sector, waypoint.x, waypoint.y)
        if waypoint_key != self.tracked_waypoint:
            self.tracked_waypoint = waypoint_key
            self.distance_history.clear()
        self.distance_history.append(distance)
        actions: set[str] = set()
        if self.recovery_steps:
            self.recovery_steps -= 1
            actions = {"move backward", "use"}
            actions.add("turn left" if self.recovery_direction > 0 else "turn right")
            actions.add("strafe right" if self.recovery_direction > 0 else "strafe left")
            return actions, {
                "mode": "navigate", "objective": self.objective[2],
                "objective_x": self.objective[0], "objective_y": self.objective[1],
                "waypoint_x": waypoint.x, "waypoint_y": waypoint.y,
                "waypoint_special": waypoint.special, "waypoint_distance": round(distance, 2),
                "waypoints_remaining": len(self.route), "aim_error_degrees": round(error, 2),
                "stuck_recovery": True,
            }
        if error > 5.0:
            actions.add("turn left")
        elif error < -5.0:
            actions.add("turn right")
        if abs(error) < 18.0:
            actions.add("move forward")

        stuck = False
        if len(self.last_positions) == self.last_positions.maxlen:
            start = self.last_positions[0]
            progress = math.hypot(player.position_x - start[0], player.position_y - start[1])
            stuck = progress < 20.0
        if len(self.distance_history) == self.distance_history.maxlen:
            # Circling or sliding along a wall can cover plenty of ground while
            # making no progress toward the portal. Measure the goal directly.
            stuck = stuck or self.distance_history[-1] >= self.distance_history[0] - 12.0
        needs_use = waypoint.special in DOOR_SPECIALS or self.objective[2] == "exit"
        # Doom USE is harmless outside activation range. Start holding it on
        # the approach so fast multi-tic movement cannot stop against a door
        # before the next controller observation.
        if needs_use and distance < 520.0:
            actions.add("use")
        if stuck:
            actions = {"move backward", "use"}
            self.recovery_direction = self.recovery_sign
            actions.add("turn left" if self.recovery_direction > 0 else "turn right")
            actions.add("strafe right" if self.recovery_direction > 0 else "strafe left")
            self.recovery_steps = 11
            self.recovery_sign *= -1
            self.last_positions.clear()
            self.distance_history.clear()
        if not actions:
            actions.add("move forward")
        return actions, {
            "mode": "navigate", "objective": self.objective[2],
            "objective_x": self.objective[0], "objective_y": self.objective[1],
            "waypoint_x": waypoint.x, "waypoint_y": waypoint.y,
            "waypoint_special": waypoint.special, "waypoint_distance": round(distance, 2),
            "waypoints_remaining": len(self.route), "aim_error_degrees": round(error, 2),
            "stuck_recovery": stuck,
        }


def run_map(wad: Path, map_name: str, seed: int, visible: bool,
            timeout_seconds: float, retries: int, writer, trace_file,
            pace: bool = True, skill: int = 1) -> dict:
    truth = WadMap(wad, map_name)
    attempts = []
    total_video_tics = total_video_frames = 0
    for attempt in range(1, retries + 1):
        game = make_campaign_game(wad, map_name, seed + attempt, visible,
                                  timeout_seconds, skill)
        combat = TacticalOracle()
        navigation = CampaignNavigator(truth)
        counts: Counter[str] = Counter()
        started = time.perf_counter()
        last_log = started
        game.new_episode()
        combat.reset(float(game.get_game_variable(vzd.GameVariable.HEALTH)))
        steps = 0
        try:
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break
                frame = np.ascontiguousarray(state.screen_buffer)
                combat_actions, combat_meta = combat.decide(state, game)
                if combat_meta.get("line_of_sight") or combat_meta.get("under_fire"):
                    actions, telemetry = combat_actions, combat_meta
                else:
                    actions, telemetry = navigation.decide(
                        state,
                        float(game.get_game_variable(vzd.GameVariable.HEALTH)),
                        float(game.get_game_variable(vzd.GameVariable.ARMOR)),
                    )
                for action in actions:
                    counts[action] += 1
                reward = float(game.make_action(action_vector_for_names(actions, game), TICS_PER_ACTION))
                row = {
                    **provenance(),
                    "map": map_name, "attempt": attempt, "step": steps,
                    "tic": int(state.tic), "actions": sorted(actions), "reward": reward,
                    "health": float(game.get_game_variable(vzd.GameVariable.HEALTH)),
                    "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                    **telemetry,
                }
                if trace_file:
                    trace_file.write(json.dumps(row) + "\n")
                if time.perf_counter() - last_log >= 10:
                    print(json.dumps(row), flush=True)
                    last_log = time.perf_counter()
                total_video_tics += TICS_PER_ACTION
                target_frames = round(total_video_tics * 30 / TICS_PER_SECOND)
                if writer:
                    while total_video_frames < target_frames:
                        writer.append_data(frame)
                        total_video_frames += 1
                steps += 1
                if pace:
                    time.sleep(TICS_PER_ACTION / TICS_PER_SECOND)
        finally:
            dead = game.is_player_dead()
            timed_out = game.get_episode_time() >= round(timeout_seconds * TICS_PER_SECOND)
            result = {
                **provenance(),
                "map": map_name, "attempt": attempt, "completed": not dead and not timed_out,
                "dead": dead, "timed_out": timed_out, "steps": steps,
                "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                "items": int(game.get_game_variable(vzd.GameVariable.ITEMCOUNT)),
                "secrets": int(game.get_game_variable(vzd.GameVariable.SECRETCOUNT)),
                "actions": dict(counts), "wall_seconds": time.perf_counter() - started,
            }
            attempts.append(result)
            game.close()
        print(json.dumps(result), flush=True)
        if result["completed"]:
            break
    return {"map": map_name, "completed": any(row["completed"] for row in attempts),
            "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wad", type=Path, required=True)
    parser.add_argument("--maps", nargs="+", default=["E1M1"])
    parser.add_argument("--seconds-per-map", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--skill", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--no-pace", action="store_true",
                        help="Run faster than real time for admission testing.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    wad = args.wad.resolve()
    writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8) if args.output else None
    trace_file = args.trace.open("w", encoding="utf-8") if args.trace else None
    results = []
    try:
        for index, map_name in enumerate(args.maps):
            result = run_map(wad, map_name.upper(), args.seed + index * 100,
                             args.visible, args.seconds_per_map, args.retries,
                             writer, trace_file, pace=not args.no_pace,
                             skill=args.skill)
            results.append(result)
            if not result["completed"]:
                break
    finally:
        if writer:
            writer.close()
        if trace_file:
            trace_file.close()
    summary = {
        **provenance(),
        "wad": str(wad), "maps_requested": [name.upper() for name in args.maps],
        "maps_completed": sum(row["completed"] for row in results), "results": results,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
