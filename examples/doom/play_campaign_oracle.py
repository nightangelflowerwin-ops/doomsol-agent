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
from jevlike.watermark import stamp_frame
from environment import TICS_PER_ACTION, TICS_PER_SECOND, action_vector_for_names
from play_tactical_oracle import TacticalOracle, angle_delta, bearing
from wad_navigation import DOOR_SPECIALS, Portal, WadMap


BUTTONS = (
    vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK, vzd.Button.USE, vzd.Button.JUMP,
    vzd.Button.MOVE_LEFT_RIGHT_DELTA,
)
KEY_NAMES = {"BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull"}
LIFT_SPECIALS = {62}
HEALTH_NAMES = {"Medikit", "Stimpack", "HealthBonus", "Berserk"}
ARMOR_NAMES = {"GreenArmor", "BlueArmor", "ArmorBonus"}
WEAPON_NAMES = {"Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher",
                "PlasmaRifle", "BFG9000", "Chainsaw"}
AMMO_NAMES = {"Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo",
              "RocketBox", "Cell", "CellPack", "Backpack"}
POWERUP_NAMES = {"Soulsphere", "Megasphere", "BlurSphere", "InvulnerabilitySphere",
                 "RadiationSuit", "ComputerMap", "LightAmp", "Berserk"}


def episode_maps(episode: str) -> list[str]:
    """Return the mandatory ordered missions for one classic Doom episode."""
    episode = episode.upper()
    if episode not in {"E1", "E2", "E3", "E4"}:
        raise ValueError(f"unsupported episode: {episode}")
    return [f"{episode}M{mission}" for mission in range(1, 9)]


def gate_summary(requested: list[str], results: list[dict], episode: str | None) -> dict:
    completed = sum(bool(row["completed"]) for row in results)
    blocked_at = next((row["map"] for row in results if not row["completed"]), None)
    all_passed = len(results) == len(requested) and completed == len(requested)
    return {
        "gate_status": "complete" if all_passed else "blocked",
        "blocked_at": blocked_at or (requested[len(results)] if len(results) < len(requested) else None),
        "missions_attempted": [row["map"] for row in results],
        "missions_completed": completed,
        "episode": episode,
        "episode_completed": bool(episode and all_passed),
    }


def has_valid_combat_threat(combat_meta: dict) -> bool:
    """Only let authoritative, currently visible threats preempt navigation.

    Recent damage is useful evidence for an evasive response, but it is not
    enough to keep tracking a map-known enemy through walls or across floors.
    """
    return bool(combat_meta.get("line_of_sight") or
                int(combat_meta.get("visible_enemies", 0)) > 0)


def gate_combat_is_urgent(combat_meta: dict) -> bool:
    """Allow combat to interrupt a mission gate only for immediate danger."""
    if combat_meta.get("damaged"):
        return True
    return bool(combat_meta.get("line_of_sight") and
                float(combat_meta.get("target_distance", float("inf"))) <= 256.0)


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
    game.set_button_max_value(vzd.Button.MOVE_LEFT_RIGHT_DELTA, 8.0)
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
    def __init__(self, map_truth: WadMap, semantic_memory: dict | None = None) -> None:
        self.map = map_truth
        self.objective: tuple[float, float, str] | None = None
        self.objective_steps = 0
        self.route: deque[Portal] = deque()
        self.route_source_sector: int | None = None
        self.last_positions: deque[tuple[float, float]] = deque(maxlen=24)
        self.distance_history: deque[float] = deque(maxlen=24)
        self.tracked_waypoint: tuple[int, float, float] | None = None
        self.recovery_sign = 1
        self.recovery_direction = 1
        self.recovery_steps = 0
        self.gate_commit_steps = 0
        self.gate_commit_key: tuple[int, float, float] | None = None
        self.gate_commit_heading: float | None = None
        self.gate_target_seen_frames = 0
        self.gate_source_sector: int | None = None
        self.transition_guard: tuple[int, int, float, float] | None = None
        self.blocked_edges: set[tuple[int, int, float, float]] = set()
        self.portal_stage_key: tuple[int, float, float] | None = None
        self.portal_stage_steps = 0
        self.mission_portal_retries: Counter[tuple[int, float, float]] = Counter()
        self.waypoint_stall_cycles: Counter[tuple[int, float, float]] = Counter()
        self.activated_lifts: set[tuple[float, float, int]] = set()
        self.activated_switches: set[tuple[float, float, int, int, float, float]] = set()
        self.lift_wait_steps = 0
        self.lift_transaction_steps = 0
        self.lift_transaction_source: int | None = None
        self.detour_key: tuple[int, float, float] | None = None
        self.detour_points: deque[tuple[float, float]] = deque()
        self.detour_transaction_target: int | None = None
        self.detour_transaction_steps = 0
        self.detour_clearance_replan_used = False
        self.portal_jump_steps = 0
        self.portal_jump_used = False
        self.precision_gap_steps = 0
        self.initial_sector_floors: tuple[float, ...] | None = None
        self.scan_steps = 0
        self.scan_reason: str | None = None
        self.stuck_events = 0
        self.breadcrumbs: deque[tuple[float, float, int | None]] = deque(maxlen=256)
        self.backtrack_target: tuple[float, float, str] | None = None
        self.ignored_pickups: set[tuple[int, int, str]] = set()
        self.ignored_switches: set[tuple[float, float, int, int, float, float]] = set()
        self.loot_sweep_steps = 0
        self.semantic_memory = semantic_memory or {}
        self.sector_visits: Counter[int] = Counter()
        self.blocked_points = [
            (float(item["x"]), float(item["y"]))
            for item in self.semantic_memory.get("stall_hotspots", [])
            if int(item.get("observations", 0)) >= 2
        ]

    def request_scan(self, reason: str) -> None:
        """Schedule one full survey before committing to the next route."""
        if not self.scan_steps:
            self.scan_steps = 18
            self.scan_reason = reason
            if reason == "room_cleared":
                self.loot_sweep_steps = 120

    def _remember_position(self, player) -> None:
        point = (float(player.position_x), float(player.position_y))
        sector = self.map.sector_at(*point)
        if sector is not None:
            self.sector_visits[sector] += 1
        if not self.breadcrumbs:
            self.breadcrumbs.append((*point, sector))
            return
        previous = self.breadcrumbs[-1]
        if sector != previous[2] or math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 96:
            self.breadcrumbs.append((*point, sector))

    def _begin_backtrack(self, player) -> None:
        """Return to the latest meaningfully different room/door approach."""
        while self.breadcrumbs:
            x, y, _ = self.breadcrumbs.pop()
            if math.hypot(x - player.position_x, y - player.position_y) >= 160:
                self.backtrack_target = (x, y, "breadcrumb_backtrack")
                self.objective = None
                self.route.clear()
                return

    def _choose_objective(self, state, player, health: float,
                          armor: float) -> tuple[float, float, str]:
        if self.backtrack_target is not None:
            distance = math.hypot(self.backtrack_target[0] - player.position_x,
                                  self.backtrack_target[1] - player.position_y)
            if distance > 48:
                return self.backtrack_target
            self.backtrack_target = None
            self.stuck_events = 0
        pending_switches = [switch for switch in self.map.switch_points
                            if switch not in self.activated_switches and
                            switch not in self.ignored_switches]
        keys = [item for item in (state.objects or []) if item.name in KEY_NAMES]
        if health <= 30:
            emergency_supplies = [
                item for item in (state.objects or []) if item.name in HEALTH_NAMES and
                (round(item.position_x), round(item.position_y), item.name)
                not in self.ignored_pickups
            ]
            if emergency_supplies:
                item = min(emergency_supplies, key=lambda value: math.hypot(
                    value.position_x - player.position_x,
                    value.position_y - player.position_y,
                ))
                if math.hypot(item.position_x - player.position_x,
                              item.position_y - player.position_y) <= 800.0:
                    return float(item.position_x), float(item.position_y), item.name
        weapons = [item for item in (state.objects or []) if item.name in WEAPON_NAMES
                   and (round(item.position_x), round(item.position_y), item.name)
                   not in self.ignored_pickups]
        if weapons:
            item = min(weapons, key=lambda value: math.hypot(
                value.position_x - player.position_x, value.position_y - player.position_y
            ))
            # Weapons are valuable, but a distant pickup must not repeatedly
            # preempt the key/exit route. Room-clear sweeps may still collect a
            # nearby upgrade before mission navigation resumes.
            weapon_limit = 480 if self.loot_sweep_steps else 220
            if math.hypot(item.position_x - player.position_x,
                          item.position_y - player.position_y) < weapon_limit:
                return float(item.position_x), float(item.position_y), item.name
        if keys and health > 30:
            key = min(keys, key=lambda item: math.hypot(
                item.position_x - player.position_x, item.position_y - player.position_y
            ))
            # A nearby key is direct, visible progression evidence. It must
            # outrank speculative traversal to every remaining map switch.
            # The route planner will still resolve any genuinely required
            # blocker encountered on the way.
            if math.hypot(key.position_x - player.position_x,
                          key.position_y - player.position_y) <= 480.0:
                return float(key.position_x), float(key.position_y), key.name
        # Mission progression outranks optional loot after the bounded
        # room-clear sweep.  During that sweep, the nearby weapon branch above
        # deliberately runs first so drops/crate upgrades are not abandoned.
        if keys and health >= 55 and not pending_switches:
            key = min(keys, key=lambda item: math.hypot(
                item.position_x - player.position_x, item.position_y - player.position_y
            ))
            return float(key.position_x), float(key.position_y), key.name
        if self.loot_sweep_steps:
            powerups = [item for item in (state.objects or []) if item.name in POWERUP_NAMES
                        and (round(item.position_x), round(item.position_y), item.name)
                        not in self.ignored_pickups]
            if powerups:
                item = min(powerups, key=lambda value: math.hypot(
                    value.position_x - player.position_x, value.position_y - player.position_y
                ))
                if math.hypot(item.position_x - player.position_x,
                              item.position_y - player.position_y) < 1000:
                    return float(item.position_x), float(item.position_y), item.name
        if health < 55:
            supplies = [item for item in (state.objects or []) if item.name in HEALTH_NAMES
                        and (round(item.position_x), round(item.position_y), item.name)
                        not in self.ignored_pickups]
            if supplies:
                item = min(supplies, key=lambda value: math.hypot(
                    value.position_x - player.position_x, value.position_y - player.position_y
                ))
                supply_distance = math.hypot(item.position_x - player.position_x,
                                             item.position_y - player.position_y)
                # Mild damage never justifies abandoning nearby mandatory
                # progression for a bonus across the map. Critical health is
                # allowed a wider, still bounded survival detour.
                supply_limit = 800.0 if health <= 30 else 320.0
                if supply_distance <= supply_limit:
                    return float(item.position_x), float(item.position_y), item.name
        # Operate map-authored progression switches before chasing keys behind
        # their tagged blockers. Immediate weapons and emergency health above
        # deliberately outrank progression so the agent survives the route.
        if pending_switches:
            x, y, special, tag, _, _ = min(
                pending_switches,
                key=lambda switch: math.hypot(switch[0] - player.position_x,
                                               switch[1] - player.position_y),
            )
            return x, y, f"switch:{special}:{tag}"
        if armor < 20:
            supplies = [item for item in (state.objects or []) if item.name in ARMOR_NAMES
                        and (round(item.position_x), round(item.position_y), item.name)
                        not in self.ignored_pickups]
            if supplies:
                item = min(supplies, key=lambda value: math.hypot(
                    value.position_x - player.position_x, value.position_y - player.position_y
                ))
                limit = 96 if item.name == "ArmorBonus" else 300
                if math.hypot(item.position_x - player.position_x,
                              item.position_y - player.position_y) < limit:
                    return float(item.position_x), float(item.position_y), item.name
        # Survival surface area: upgrade firepower and replenish ammunition
        # before a distant key/exit, but do not cross the whole map for scraps.
        survival_items = [item for item in (state.objects or [])
                          if item.name in AMMO_NAMES
                          and (round(item.position_x), round(item.position_y), item.name)
                          not in self.ignored_pickups]
        if survival_items:
            def survival_score(item) -> float:
                distance = math.hypot(item.position_x - player.position_x,
                                      item.position_y - player.position_y)
                return 160.0 - distance
            item = max(survival_items, key=survival_score)
            distance = math.hypot(item.position_x - player.position_x,
                                  item.position_y - player.position_y)
            if survival_score(item) > 0 and distance < 160:
                return float(item.position_x), float(item.position_y), item.name
        if keys:
            key = min(keys, key=lambda item: math.hypot(
                item.position_x - player.position_x, item.position_y - player.position_y
            ))
            return float(key.position_x), float(key.position_y), key.name
        x, y, _ = self.map.exit_points[0]
        return x, y, "exit"

    def decide(self, state, health: float, armor: float) -> tuple[set[str], dict]:
        player = next(item for item in state.objects if item.name == "DoomPlayer")
        self._remember_position(player)
        mandatory_key_latched = bool(
            self.objective is not None and self.objective[2] in KEY_NAMES and
            any(item.name == self.objective[2] and
                round(item.position_x) == round(self.objective[0]) and
                round(item.position_y) == round(self.objective[1])
                for item in (state.objects or []))
        )
        if mandatory_key_latched:
            # A visible key is an explicit progression transaction.  Optional
            # room-clear/navigation surveys must never turn the player away
            # from it once the exact pickup has been selected.
            self.scan_steps = 0
            self.scan_reason = None
        if health <= 30.0:
            # A survey is never worth dying for. Abort spins and stale gate
            # transactions so emergency health can be selected immediately.
            self.scan_steps = 0
            self.scan_reason = None
            self.gate_commit_key = None
            self.gate_commit_steps = 0
            self.detour_transaction_target = None
        if self.loot_sweep_steps:
            self.loot_sweep_steps -= 1
        if self.scan_steps:
            self.scan_steps -= 1
            reason = self.scan_reason
            if not self.scan_steps:
                self.scan_reason = None
            return {"turn left"}, {
                "mode": "survey_360", "scan_reason": reason,
                "scan_steps_remaining": self.scan_steps,
                "breadcrumbs": len(self.breadcrumbs),
                "objective": self.objective[2] if self.objective else None,
                "stuck_recovery": False,
            }
        # Once an in-range gate commit begins, do not let newly visible ammo,
        # armor, or weapons replace the route before the sector crossing.
        pickup_latched = bool(
            self.objective is not None and self.objective[2] in WEAPON_NAMES and
            (round(self.objective[0]), round(self.objective[1]), self.objective[2])
            not in self.ignored_pickups and
            any(item.name == self.objective[2] and
                round(item.position_x) == round(self.objective[0]) and
                round(item.position_y) == round(self.objective[1])
                for item in (state.objects or []))
        )
        key_latched = bool(
            self.objective is not None and self.objective[2] in KEY_NAMES and
            any(item.name == self.objective[2] and
                round(item.position_x) == round(self.objective[0]) and
                round(item.position_y) == round(self.objective[1])
                for item in (state.objects or []))
        )
        if health <= 30.0:
            current_objective = self._choose_objective(state, player, health, armor)
        elif pickup_latched or key_latched:
            # A visible crate/dropped weapon is a bounded transaction. Keep
            # its exact object identity until collection or the existing
            # three-cycle recovery marks it unreachable; do not alternate
            # between another weapon and the mission switch every frame. Keys
            # are mandatory progression transactions and remain latched until
            # the exact object disappears at the pickup coordinate.
            current_objective = self.objective
        elif (self.objective is not None and self.objective[2] == "exit" and
                health > 30.0):
            # Once progression has reached the exit phase, optional armor,
            # Berserk, ammo, and distant weapons must not steal the remaining
            # mission budget. Only critical survival may break this latch.
            current_objective = self.objective
        elif ((self.gate_commit_key is not None or
             self.detour_transaction_target is not None) and
                self.objective is not None):
            current_objective = self.objective
        else:
            current_objective = self._choose_objective(state, player, health, armor)
        objective_changed = self.objective != current_objective
        if objective_changed:
            self.objective_steps = 0
        else:
            self.objective_steps += 1
        if (current_objective[2] in WEAPON_NAMES and
                self.objective_steps >= 120):
            # An optional pickup is never allowed to consume the mission. If
            # it remains present after a bounded transaction, the route or
            # crate is currently unreachable. Ignore that exact instance and
            # immediately return to progression.
            self.ignored_pickups.add((round(current_objective[0]),
                                      round(current_objective[1]),
                                      current_objective[2]))
            current_objective = self._choose_objective(state, player, health, armor)
            objective_changed = True
            self.objective_steps = 0
            self.route.clear()
        if (current_objective[2].startswith("switch:") and
                self.objective_steps >= 240):
            _, special_text, tag_text = current_objective[2].split(":")
            matching = next((switch for switch in self.map.switch_points
                             if round(switch[0]) == round(current_objective[0]) and
                             round(switch[1]) == round(current_objective[1]) and
                             switch[2] == int(special_text) and
                             switch[3] == int(tag_text)), None)
            if matching is not None:
                self.ignored_switches.add(matching)
            current_objective = self._choose_objective(state, player, health, armor)
            objective_changed = True
            self.objective_steps = 0
            self.route.clear()
        self.objective = current_objective
        if self.initial_sector_floors is None:
            self.initial_sector_floors = tuple(
                float(sector.floor_height) for sector in state.sectors
            )
        self.last_positions.append((player.position_x, player.position_y))
        current_sector = self.map.sector_at(player.position_x, player.position_y)

        tag3_is_lowered = any(
            switch[3] == 3 for switch in self.activated_switches
        )

        if self.precision_gap_steps > 0:
            if float(player.position_y) >= 32.0:
                self.precision_gap_steps = 0
                self.last_positions.clear()
                self.distance_history.clear()
            else:
                self.precision_gap_steps -= 1
                x_error = 2352.0 - float(player.position_x)
                heading_error = angle_delta(90.0, float(player.angle))
                precision_actions: set[str] = set()
                if heading_error > 2.0:
                    precision_actions.add("turn left")
                elif heading_error < -2.0:
                    precision_actions.add("turn right")
                elif abs(x_error) > 0.35:
                    # Digital strafe moves several units per tic and
                    # oscillates across this player-width gap. The execution
                    # layer applies this bounded analogue correction below.
                    pass
                else:
                    precision_actions.update({"move forward", "jump"})
                return precision_actions, {
                    "mode": "precision_gap_crossing",
                    "objective": self.objective[2],
                    "current_sector": current_sector,
                    "gate_target_sector": 119,
                    "player_x": round(float(player.position_x), 3),
                    "player_y": round(float(player.position_y), 3),
                    "player_angle": round(float(player.angle), 3),
                    "gap_center_x": 2352.0,
                    "x_error": round(x_error, 3),
                    "lateral_delta": round(max(-3.0, min(3.0, x_error)), 3),
                    "precision_steps_remaining": self.precision_gap_steps,
                    "action_tics": 1,
                    "stuck_recovery": True,
                }

        if (current_sector == 134 and current_objective[2] == "BlueCard" and
                not tag3_is_lowered):
            for portal in self.map.graph[134]:
                if portal.target_sector == 120:
                    self.blocked_edges.add((
                        134, 120, float(portal.x), float(portal.y)
                    ))

        if self.detour_transaction_target is not None:
            if current_sector == self.detour_transaction_target:
                self.detour_transaction_target = None
                self.detour_transaction_steps = 0
            elif self.detour_transaction_steps > 0:
                self.detour_transaction_steps -= 1
            else:
                self.detour_transaction_target = None

        if self.lift_transaction_steps > 0:
            if current_sector != self.lift_transaction_source:
                self.lift_transaction_steps = 0
                self.lift_transaction_source = None
            else:
                self.lift_transaction_steps -= 1

        if self.transition_guard is not None:
            source_sector, target_sector, portal_x, portal_y = self.transition_guard
            if current_sector == source_sector:
                self.blocked_edges.add(self.transition_guard)
                self.transition_guard = None
                self.route.clear()
                self.route_source_sector = current_sector
                self.last_positions.clear()
                self.distance_history.clear()
                return set(), {
                    "mode": "transition_regression",
                    "objective": self.objective[2],
                    "current_sector": current_sector,
                    "blocked_edge_source": source_sector,
                    "blocked_edge_target": target_sector,
                    "blocked_edge_x": portal_x,
                    "blocked_edge_y": portal_y,
                    "stuck_recovery": False,
                }
            if current_sector not in {source_sector, target_sector}:
                self.transition_guard = None

        if self.gate_commit_key is not None:
            committed_target = self.gate_commit_key[0]
            planned_targets = [portal.target_sector for portal in list(self.route)[:-1]]
            target_index = (planned_targets.index(committed_target)
                            if committed_target in planned_targets else None)
            current_index = (planned_targets.index(current_sector)
                             if current_sector in planned_targets else None)
            reached_commit = bool(
                current_sector == committed_target or
                (target_index is not None and current_index is not None and
                 current_index >= target_index)
            )
            if reached_commit:
                self.gate_target_seen_frames += 1
                if self.gate_target_seen_frames < 2:
                    return set(), {
                        "mode": "gate_confirm", "objective": self.objective[2],
                        "gate_target_sector": committed_target,
                        "current_sector": current_sector,
                        "gate_target_seen_frames": self.gate_target_seen_frames,
                        "player_x": round(float(player.position_x), 2),
                        "player_y": round(float(player.position_y), 2),
                        "stuck_recovery": False,
                    }
                if self.gate_source_sector is not None:
                    self.transition_guard = (
                        self.gate_source_sector,
                        committed_target,
                        float(self.gate_commit_key[1]),
                        float(self.gate_commit_key[2]),
                    )
                self.gate_commit_key = None
                self.gate_commit_steps = 0
                self.gate_commit_heading = None
                self.gate_target_seen_frames = 0
                self.gate_source_sector = None
            else:
                self.gate_target_seen_frames = 0
                if self.gate_commit_steps <= 0:
                    failed_key = self.gate_commit_key
                    self.mission_portal_retries[failed_key] += 1
                    retry_count = self.mission_portal_retries[failed_key]
                    if retry_count >= 3:
                        failed_portal = (float(failed_key[1]), float(failed_key[2]))
                        if failed_portal not in self.blocked_points:
                            self.blocked_points.append(failed_portal)
                        self.route.clear()
                        self.route_source_sector = current_sector
                    self.gate_commit_key = None
                    self.gate_commit_heading = None
                    self.portal_stage_key = None
                    self.portal_stage_steps = 0
                    self.gate_source_sector = None
                    self.last_positions.clear()
                    self.distance_history.clear()
                    self.stuck_events = 0
                    return set(), {
                        "mode": "gate_failed", "objective": self.objective[2],
                        "failed_portal_x": float(failed_key[1]),
                        "failed_portal_y": float(failed_key[2]),
                        "gate_target_sector": committed_target,
                        "current_sector": current_sector,
                        "mission_portal_retries": retry_count,
                        "portal_blocked": retry_count >= 3,
                        "stuck_recovery": False,
                    }

        route_targets = [portal.target_sector for portal in list(self.route)[:-1]]
        detour_transit = bool(
            (self.detour_transaction_target == 119 and
             current_sector in {134, 120} and not tag3_is_lowered) or
            (current_sector == 120 and 119 in route_targets)
        )
        if (self.route and current_sector != self.route_source_sector and
                current_sector not in route_targets and not detour_transit):
            # The live engine state is authoritative. Never retain a route
            # whose first edge belongs to a sector the player has already left.
            self.route.clear()

        if objective_changed or not self.route:
            self.route = deque(self.map.route(
                (player.position_x, player.position_y), current_objective[:2],
                blocked_points=self.blocked_points,
                blocked_edges=self.blocked_edges,
                unlocked_tags={switch[3] for switch in self.activated_switches},
            ))
            if (not self.route and
                    current_objective[2].startswith("switch:")):
                # A failed timed-lift crossing is not proof that the only
                # progression route is impossible. Re-arm the lift and retry
                # the authoritative route instead of degrading to a straight
                # wall-pushing vector toward the switch.
                self.blocked_points.clear()
                self.blocked_edges.clear()
                self.activated_lifts.clear()
                self.route = deque(self.map.route(
                    (player.position_x, player.position_y), current_objective[:2],
                    unlocked_tags={switch[3] for switch in self.activated_switches},
                ))
            self.route_source_sector = current_sector

        if self.lift_wait_steps > 0:
            self.lift_wait_steps -= 1
            return set(), {
                "mode": "lift_wait", "objective": self.objective[2],
                "current_sector": current_sector,
                "lift_wait_steps": self.lift_wait_steps,
                "stuck_recovery": False,
            }

        # Some raised platforms are surrounded by a rim narrower than the
        # player's collision diameter. Their lift action must be triggered
        # from the surrounding sector before the first rim portal is walkable.
        nearby_route = list(self.route)[:2]
        lift_portal = next((
            portal for portal in nearby_route
            if portal.special in LIFT_SPECIALS and portal.floor_delta > 32 and
            math.hypot(portal.x - player.position_x,
                       portal.y - player.position_y) <= 96.0 and
            (float(portal.x), float(portal.y), portal.special) not in self.activated_lifts
        ), None)
        if lift_portal is not None:
            # The routed portal midpoint is not necessarily a reachable USE
            # face.  E1M1's raised platform has several special-62 linedefs;
            # one midpoint sits inside its collision rim.  Resolve the lift's
            # source sector, then approach the nearest equivalent linedef from
            # a probe that is authoritatively in the player's current sector.
            lift_source = next((
                source for source, portals in self.map.graph.items()
                if lift_portal in portals
            ), None)
            activation_portal = lift_portal
            activation_x = float(lift_portal.x)
            activation_y = float(lift_portal.y)
            if lift_source is not None:
                candidates = []
                for candidate in self.map.graph[lift_source]:
                    if (candidate.special != lift_portal.special or
                            candidate.floor_delta <= 32):
                        continue
                    probe_x = candidate.x - candidate.normal_x * 24.0
                    probe_y = candidate.y - candidate.normal_y * 24.0
                    if self.map.sector_at(probe_x, probe_y) != current_sector:
                        continue
                    candidates.append((
                        math.hypot(probe_x - player.position_x,
                                   probe_y - player.position_y),
                        candidate, probe_x, probe_y,
                    ))
                if candidates:
                    _, activation_portal, activation_x, activation_y = min(
                        candidates, key=lambda item: item[0]
                    )
            lift_target = type("LiftTarget", (), {
                "position_x": activation_x, "position_y": activation_y
            })()
            lift_heading = bearing(player, lift_target)
            lift_error = angle_delta(lift_heading, float(player.angle))
            lift_distance = math.hypot(
                activation_x - player.position_x,
                activation_y - player.position_y,
            )
            # At the exterior probe, turn toward the linedef itself before
            # pressing USE.  This keeps activation local and prevents distant
            # accidental interactions.
            if lift_distance <= 32.0:
                line_target = type("LiftLineTarget", (), {
                    "position_x": activation_portal.x,
                    "position_y": activation_portal.y,
                })()
                lift_heading = bearing(player, line_target)
                lift_error = angle_delta(lift_heading, float(player.angle))
            lift_actions: set[str] = set()
            if lift_error > 5.0:
                lift_actions.add("turn left")
            elif lift_error < -5.0:
                lift_actions.add("turn right")
            if lift_distance > 32.0 and abs(lift_error) < 18.0:
                lift_actions.add("move forward")
            elif lift_distance <= 32.0 and abs(lift_error) < 22.0:
                lift_actions.add("use")
                self.activated_lifts.add((
                    float(lift_portal.x), float(lift_portal.y), lift_portal.special
                ))
                # Special 62 lowers, waits, then raises.  Twelve controller
                # decisions let the platform descend but leave most of the
                # bottom window available for the immediate portal commit.
                self.lift_wait_steps = 12
                self.lift_transaction_steps = 50
                self.lift_transaction_source = current_sector
            return lift_actions, {
                "mode": "lift_activate", "objective": self.objective[2],
                "current_sector": current_sector,
                "lift_x": lift_portal.x, "lift_y": lift_portal.y,
                "activation_x": round(activation_x, 2),
                "activation_y": round(activation_y, 2),
                "activation_line_x": activation_portal.x,
                "activation_line_y": activation_portal.y,
                "lift_special": lift_portal.special,
                "lift_distance": round(lift_distance, 2),
                "aim_error_degrees": round(lift_error, 2),
                "stuck_recovery": False,
            }
        # One controller action advances multiple Doom tics, so a narrow door
        # sector can be entered and exited between observations. If the player
        # is authoritatively observed in a later planned sector, confirm every
        # intervening portal instead of waiting forever for the thin sector.
        route_sectors = [portal.target_sector for portal in list(self.route)[:-1]]
        # Sector 120 is a thin geometric transit band alongside the concave
        # 134 -> 119 approach.  It can also occur later in the planned route,
        # so treating its transient observation as forward route progress
        # incorrectly discards the still-unreached sector-119 portal.
        preserve_detour_route = bool(
            (self.detour_transaction_target == 119 and
             current_sector in {134, 120} and not tag3_is_lowered) or
            (current_sector == 120 and 119 in route_sectors)
        )
        if current_sector in route_sectors and not preserve_detour_route:
            crossed_index = route_sectors.index(current_sector)
            for _ in range(crossed_index + 1):
                self.route.popleft()
            self.route_source_sector = current_sector
        # A portal is complete only after the player actually enters its target
        # sector. Distance to a linedef is not proof of crossing it. The final
        # synthetic waypoint is the sole exception and represents the physical
        # objective inside the destination sector.
        while len(self.route) > 1 and current_sector == self.route[0].target_sector:
            self.route.popleft()
        while len(self.route) == 1 and math.hypot(
            self.route[0].x - player.position_x,
            self.route[0].y - player.position_y,
        ) < 40.0:
            self.route.popleft()
        if (not self.route and self.detour_transaction_target == 119 and
                not tag3_is_lowered):
            # The concave approach may temporarily leave the navigation graph
            # unable to reconstruct a route while the player is touching the
            # adjacent sector-120 band.  Never degrade to the distant key
            # coordinate during this transaction: retain the authoritative
            # 134 -> 119 portal until sector 119 is actually observed.
            waypoint = next(
                portal for portal in self.map.graph[134]
                if portal.target_sector == 119
            )
        elif not self.route:
            waypoint = Portal(-1, current_objective[0], current_objective[1], 0)
        else:
            waypoint = self.route[0]

        needs_use = waypoint.special in DOOR_SPECIALS or self.objective[2] == "exit"
        is_sector_portal = waypoint.target_sector >= 0
        # Open portals need a perpendicular approach. First stage just inside
        # the authoritative source sector; the committed action then follows
        # the directed normal through the boundary.
        staged_open_portal = bool(
            is_sector_portal and not needs_use and
            (waypoint.normal_x or waypoint.normal_y) and
            self.gate_commit_key != (waypoint.target_sector, waypoint.x, waypoint.y)
        )
        aim_x, aim_y = waypoint.x, waypoint.y
        if staged_open_portal:
            aim_x = waypoint.x - waypoint.normal_x * 24.0
            aim_y = waypoint.y - waypoint.normal_y * 24.0
        waypoint_key = (waypoint.target_sector, waypoint.x, waypoint.y)
        if waypoint_key != self.tracked_waypoint:
            self.tracked_waypoint = waypoint_key
            self.distance_history.clear()
            self.detour_key = None
            self.detour_points.clear()
            if (staged_open_portal and current_sector == 134 and
                    waypoint.target_sector == 119 and not tag3_is_lowered):
                local_path = self.map.local_path(
                    (float(player.position_x), float(player.position_y)),
                    (float(aim_x), float(aim_y)), current_sector,
                    avoid_targets=(None if any(
                        switch[3] == 3 for switch in self.activated_switches
                    ) else {120}),
                )
                if len(local_path) > 2:
                    self.detour_key = waypoint_key
                    self.detour_points.extend(local_path[1:-1])
                    self.detour_transaction_target = 119
                    # Long enough for the bounded concave-wall recovery and
                    # incoming-fire evasions, but still finite if the map
                    # geometry is genuinely unreachable.  Authoritative entry
                    # into sector 119 clears this immediately above.
                    self.detour_transaction_steps = 1200
                    self.detour_clearance_replan_used = False
                    self.portal_jump_used = False
        if self.detour_key == waypoint_key:
            while self.detour_points and math.hypot(
                    self.detour_points[0][0] - player.position_x,
                    self.detour_points[0][1] - player.position_y) < 28.0:
                self.detour_points.popleft()
            if self.detour_points:
                aim_x, aim_y = self.detour_points[0]
            else:
                self.detour_key = None
        desired = bearing(player, type("Waypoint", (), {
            "position_x": aim_x, "position_y": aim_y
        })())
        error = angle_delta(desired, float(player.angle))
        distance = math.hypot(aim_x - player.position_x, aim_y - player.position_y)
        self.distance_history.append(distance)
        actions: set[str] = set()
        switch_objective_distance = math.hypot(
            current_objective[0] - player.position_x,
            current_objective[1] - player.position_y,
        )
        if current_objective[2].startswith("switch:"):
            _, special_text, tag_text = current_objective[2].split(":")
            switch = next(item for item in self.map.switch_points if
                          item[0] == float(current_objective[0]) and
                          item[1] == float(current_objective[1]) and
                          item[2] == int(special_text) and
                          item[3] == int(tag_text))
            tagged_sectors = [index for index, tag in enumerate(self.map.sector_tags)
                              if tag == switch[3]]
            changed_sectors = [index for index in tagged_sectors if
                               abs(float(state.sectors[index].floor_height) -
                                   self.initial_sector_floors[index]) > 4.0]
            if changed_sectors:
                self.activated_switches.add(switch)
                self.objective = None
                self.route.clear()
                self.blocked_edges.clear()
                self.blocked_points.clear()
                self.detour_key = None
                self.detour_points.clear()
                self.detour_transaction_target = None
                self.detour_transaction_steps = 0
                self.precision_gap_steps = 0
                self.tracked_waypoint = None
                self.last_positions.clear()
                self.distance_history.clear()
                return set(), {
                    "mode": "progression_switch_confirmed",
                    "objective": current_objective[2],
                    "switch_activated": True,
                    "changed_sectors": changed_sectors,
                    "stuck_recovery": False,
                }
            switch_target = type("SwitchTarget", (), {
                "position_x": switch[4], "position_y": switch[5]
            })()
            switch_line_distance = math.hypot(
                switch[4] - player.position_x, switch[5] - player.position_y
            )
            if switch_line_distance > 56.0:
                # Continue the routed approach until Doom's USE range reaches
                # the actual linedef, not merely the nearby planning point.
                pass
            else:
                switch_error = angle_delta(bearing(player, switch_target),
                                           float(player.angle))
                actions.clear()
                if switch_error > 5.0:
                    actions.add("turn left")
                elif switch_error < -5.0:
                    actions.add("turn right")
                else:
                    actions.add("use")
                    if switch_line_distance > 36.0:
                        actions.add("move forward")
                return actions, {
                    "mode": "progression_switch", "objective": current_objective[2],
                    "objective_x": current_objective[0], "objective_y": current_objective[1],
                    "player_x": round(float(player.position_x), 2),
                    "player_y": round(float(player.position_y), 2),
                    "player_angle": round(float(player.angle), 2),
                    "switch_distance": round(switch_objective_distance, 2),
                    "switch_line_distance": round(switch_line_distance, 2),
                    "switch_activated": False,
                    "switch_use_attempted": "use" in actions,
                    "stuck_recovery": False,
                }
        if self.portal_jump_steps > 0:
            self.portal_jump_steps -= 1
            jump_error = angle_delta(90.0, float(player.angle))
            jump_actions = {"move forward", "jump"}
            if jump_error > 5.0:
                jump_actions.add("turn left")
            elif jump_error < -5.0:
                jump_actions.add("turn right")
            return jump_actions, {
                "mode": "portal_jump_commit",
                "objective": self.objective[2],
                "current_sector": current_sector,
                "gate_target_sector": 119,
                "player_x": round(float(player.position_x), 2),
                "player_y": round(float(player.position_y), 2),
                "player_angle": round(float(player.angle), 2),
                "jump_steps_remaining": self.portal_jump_steps,
                "red_light_jump_trigger": True,
                "stuck_recovery": True,
            }
        if self.gate_commit_key == waypoint_key and distance > 240.0:
            self.gate_commit_key = None
            self.gate_commit_steps = 0
            self.gate_commit_heading = None
            self.gate_target_seen_frames = 0
        recent_distances = list(self.distance_history)[-8:]
        if (current_objective[2] in WEAPON_NAMES and distance <= 96.0 and
                len(recent_distances) == 8 and
                max(recent_distances) - min(recent_distances) < 4.0):
            self.waypoint_stall_cycles[waypoint_key] += 1
            pickup_attempt = self.waypoint_stall_cycles[waypoint_key]
            self.last_positions.clear()
            self.distance_history.clear()
            if pickup_attempt <= 3:
                pickup_actions = {"attack", "use", "jump"}
                if error > 5.0:
                    pickup_actions.add("turn left")
                elif error < -5.0:
                    pickup_actions.add("turn right")
                if abs(error) < 22.0:
                    pickup_actions.add("move forward")
                return pickup_actions, {
                    "mode": "crate_weapon_recovery",
                    "objective": current_objective[2],
                    "objective_x": current_objective[0],
                    "objective_y": current_objective[1],
                    "pickup_distance": round(distance, 2),
                    "pickup_attempt": pickup_attempt,
                    "stuck_recovery": True,
                }
            self.ignored_pickups.add((round(current_objective[0]),
                                      round(current_objective[1]),
                                      current_objective[2]))
            self.objective = None
            self.route.clear()
            return set(), {
                "mode": "crate_weapon_abandoned",
                "objective": current_objective[2],
                "pickup_distance": round(distance, 2),
                "pickup_attempt": pickup_attempt,
                "stuck_recovery": False,
            }
        at_portal_approach = not (
            self.detour_key == waypoint_key and self.detour_points
        )
        contact_stall = bool(
            staged_open_portal and at_portal_approach and
            len(recent_distances) == 8 and
            distance <= 52.0 and abs(error) < 18.0 and
            max(recent_distances) - min(recent_distances) < 2.0
        )
        if (staged_open_portal and at_portal_approach and
                (distance <= 32.0 or contact_stall) and
                self.portal_stage_key != waypoint_key):
            self.portal_stage_key = waypoint_key
            self.portal_stage_steps = 18
        if self.portal_stage_key == waypoint_key and self.gate_commit_key != waypoint_key:
            normal_heading = math.degrees(math.atan2(
                waypoint.normal_y, waypoint.normal_x
            )) % 360.0
            stage_error = angle_delta(normal_heading, float(player.angle))
            speed = math.hypot(
                float(getattr(player, "velocity_x", 0.0)),
                float(getattr(player, "velocity_y", 0.0)),
            )
            # One turn action advances several tics and rotates roughly 14
            # degrees. An 8-degree dead-zone is the smallest reachable stable
            # band; a 6-degree gate oscillates forever around zero.
            if abs(stage_error) <= 8.0 and speed <= 1.0:
                self.gate_commit_key = waypoint_key
                self.gate_commit_steps = 14
                self.gate_commit_heading = normal_heading
                self.gate_target_seen_frames = 0
                self.gate_source_sector = current_sector
                self.portal_stage_key = None
                self.portal_stage_steps = 0
            else:
                self.portal_stage_steps -= 1
                if self.portal_stage_steps <= 0:
                    self.portal_stage_key = None
                stage_actions: set[str] = set()
                if stage_error > 8.0:
                    stage_actions.add("turn left")
                elif stage_error < -8.0:
                    stage_actions.add("turn right")
                return stage_actions, {
                    "mode": "portal_stage", "objective": self.objective[2],
                    "waypoint_x": waypoint.x, "waypoint_y": waypoint.y,
                    "gate_target_sector": waypoint.target_sector,
                    "current_sector": current_sector,
                    "player_x": round(float(player.position_x), 2),
                    "player_y": round(float(player.position_y), 2),
                    "player_angle": round(float(player.angle), 2),
                    "portal_normal_x": round(float(waypoint.normal_x), 4),
                    "portal_normal_y": round(float(waypoint.normal_y), 4),
                    "stage_error_degrees": round(stage_error, 2),
                    "player_speed": round(speed, 3),
                    "portal_stage_steps": self.portal_stage_steps,
                    "stuck_recovery": False,
                }
        if (is_sector_portal and not staged_open_portal and distance <= 48.0 and
                self.gate_commit_key != waypoint_key):
            self.gate_commit_key = waypoint_key
            self.gate_commit_steps = 14
            self.gate_target_seen_frames = 0
            self.gate_source_sector = current_sector
            # Drive along the WAD linedef's directed source-to-target normal.
            # A midpoint or player-entry bearing can cross a different adjacent
            # boundary when several sectors meet near the same coordinate.
            target_x = waypoint.x + waypoint.normal_x * 32.0
            target_y = waypoint.y + waypoint.normal_y * 32.0
            self.gate_commit_heading = math.degrees(math.atan2(
                target_y - player.position_y,
                target_x - player.position_x,
            )) % 360.0
        if self.gate_commit_steps and self.gate_commit_key == waypoint_key:
            in_activation_zone = distance <= 48.0
            self.gate_commit_steps -= 1
            commit_error = angle_delta(
                self.gate_commit_heading if self.gate_commit_heading is not None else desired,
                float(player.angle),
            )
            if commit_error > 4.0:
                actions.add("turn left")
            elif commit_error < -4.0:
                actions.add("turn right")
            if abs(commit_error) < 22.0:
                actions.add("move forward")
            # Pulse USE once. Holding it every decision can toggle a repeatable
            # door closed again before the player crosses the threshold.
            if needs_use and in_activation_zone and self.gate_commit_steps == 13:
                actions.add("use")
            return actions, {
                "mode": "gate_commit", "objective": self.objective[2],
                "objective_x": self.objective[0], "objective_y": self.objective[1],
                "waypoint_x": waypoint.x, "waypoint_y": waypoint.y,
                "waypoint_special": waypoint.special, "waypoint_distance": round(distance, 2),
                "waypoints_remaining": len(self.route), "aim_error_degrees": round(commit_error, 2),
                "gate_target_sector": waypoint.target_sector,
                "current_sector": current_sector,
                "player_x": round(float(player.position_x), 2),
                "player_y": round(float(player.position_y), 2),
                "player_angle": round(float(player.angle), 2),
                "gate_commit_heading": round(float(self.gate_commit_heading), 2),
                "portal_normal_x": round(float(waypoint.normal_x), 4),
                "portal_normal_y": round(float(waypoint.normal_y), 4),
                "gate_commit_steps": self.gate_commit_steps,
                "sector_crossed": current_sector == waypoint.target_sector,
                "stuck_recovery": False,
            }
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
        skip_stall_scan = False
        if len(self.last_positions) == self.last_positions.maxlen:
            start = self.last_positions[0]
            progress = math.hypot(player.position_x - start[0], player.position_y - start[1])
            stuck = progress < 20.0
        if len(self.distance_history) == self.distance_history.maxlen:
            # Circling or sliding along a wall can cover plenty of ground while
            # making no progress toward the portal. Measure the goal directly.
            stuck = stuck or self.distance_history[-1] >= self.distance_history[0] - 12.0
        if (stuck and self.detour_transaction_target == 119 and
                waypoint.target_sector == 119 and
                self.detour_clearance_replan_used and
                not self.portal_jump_used and
                not any(switch[3] == 3 for switch in self.activated_switches) and
                2160.0 <= float(player.position_x) <= 2532.0 and
                -64.0 <= float(player.position_y) <= -36.0):
            # This is the raised/red-light band directly below the key-room
            # portal. Walking and backward-jump recovery cannot cross it;
            # commit a short forward jump while facing north.
            self.portal_jump_used = True
            self.precision_gap_steps = 240
            self.last_positions.clear()
            self.distance_history.clear()
            return {"turn left"}, {
                "mode": "precision_gap_crossing",
                "objective": self.objective[2],
                "current_sector": current_sector,
                "gate_target_sector": 119,
                "player_x": round(float(player.position_x), 2),
                "player_y": round(float(player.position_y), 2),
                "player_angle": round(float(player.angle), 2),
                "gap_center_x": 2352.0,
                "precision_steps_remaining": self.precision_gap_steps,
                "action_tics": 1,
                "stuck_recovery": True,
            }
        if (stuck and self.detour_transaction_target == 119 and
                waypoint.target_sector == 119 and
                not self.detour_clearance_replan_used):
            # The direct northbound approach can settle against the thin
            # sector-120 collision band about 152 units below this portal.
            # Make one bounded east-side clearance pass through points that
            # are authoritatively inside sector 134, then return to the portal
            # staging point.  Keep the transaction/objective latched.
            # The row at y=-48 consists of 32-unit solid pillars separated by
            # player-width gaps.  A diagonal approach wedges against a pillar.
            # Stage south of the gap centered at x=2352, align exactly, cross
            # it northbound, then turn toward the key-room portal.
            clearance_points = [(2352.0, -96.0), (2352.0, 32.0),
                                (2352.0, 112.0), (2352.0, 136.0),
                                (2304.0, 144.0), (2256.0, 168.0)]
            self.detour_key = waypoint_key
            self.detour_points.clear()
            self.detour_points.extend(clearance_points)
            self.detour_clearance_replan_used = True
            self.last_positions.clear()
            self.distance_history.clear()
            clearance_x, clearance_y = clearance_points[0]
            clearance_heading = math.degrees(math.atan2(
                clearance_y - player.position_y,
                clearance_x - player.position_x,
            )) % 360.0
            clearance_error = angle_delta(clearance_heading, float(player.angle))
            recovery_actions: set[str] = set()
            if clearance_error > 5.0:
                recovery_actions.add("turn left")
            elif clearance_error < -5.0:
                recovery_actions.add("turn right")
            if abs(clearance_error) < 18.0:
                recovery_actions.add("move forward")
            return recovery_actions or {"turn left"}, {
                "mode": "portal_clearance_replan",
                "objective": self.objective[2],
                "current_sector": current_sector,
                "gate_target_sector": 119,
                "player_x": round(float(player.position_x), 2),
                "player_y": round(float(player.position_y), 2),
                "clearance_x": clearance_x,
                "clearance_y": clearance_y,
                "stuck_recovery": True,
            }
        vertical_transition = ("rise" if waypoint.floor_delta > 16 else
                               "drop" if waypoint.floor_delta < -24 else "level")
        # Door/lift activation is handled by the single-pulse gate commit above.
        if vertical_transition == "rise" and distance < 180.0:
            actions.update({"move forward", "jump", "use"})
        mission_portal = False
        if stuck:
            self.stuck_events += 1
            if self.stuck_events >= 2:
                progression_objective = bool(
                    current_objective[2] in (KEY_NAMES | {"exit"}) or
                    current_objective[2].startswith("switch:")
                )
                mission_portal = (waypoint.target_sector >= 0 and
                                  current_sector != waypoint.target_sector and
                                  progression_objective and
                                  distance <= 96.0)
                if self.backtrack_target is not None:
                    # A failed retreat is not evidence that another, older
                    # breadcrumb will help. End the retreat and replan mission.
                    self.backtrack_target = None
                    self.objective = None
                    self.route.clear()
                elif mission_portal:
                    retry_key = (waypoint.target_sector,
                                 float(waypoint.x), float(waypoint.y))
                    self.mission_portal_retries[retry_key] += 1
                    self.gate_commit_key = None
                    self.gate_commit_steps = 0
                    self.gate_commit_heading = None
                    self.stuck_events = 0
                    if self.mission_portal_retries[retry_key] >= 3:
                        failed_portal = (float(waypoint.x), float(waypoint.y))
                        if failed_portal not in self.blocked_points:
                            self.blocked_points.append(failed_portal)
                        # Replan locally around this exact failed opening. A
                        # long breadcrumb retreat can carry the player away
                        # from another entrance into the same target sector.
                        self.route.clear()
                        self.backtrack_target = None
                        self.route_source_sector = current_sector
                        skip_stall_scan = True
                elif current_objective[2] in WEAPON_NAMES:
                    # A weapon still visible at close range may be sitting in
                    # or behind a breakable crate.  Do not immediately mark it
                    # ignored.  Try a bounded face/attack/use/jump interaction
                    # and only abandon after three verified no-progress cycles.
                    self.waypoint_stall_cycles[waypoint_key] += 1
                    pickup_attempt = self.waypoint_stall_cycles[waypoint_key]
                    self.stuck_events = 0
                    self.last_positions.clear()
                    self.distance_history.clear()
                    if pickup_attempt <= 3:
                        pickup_actions = {"attack", "use", "jump"}
                        if error > 5.0:
                            pickup_actions.add("turn left")
                        elif error < -5.0:
                            pickup_actions.add("turn right")
                        if abs(error) < 22.0:
                            pickup_actions.add("move forward")
                        return pickup_actions, {
                            "mode": "crate_weapon_recovery",
                            "objective": current_objective[2],
                            "objective_x": current_objective[0],
                            "objective_y": current_objective[1],
                            "player_x": round(float(player.position_x), 2),
                            "player_y": round(float(player.position_y), 2),
                            "pickup_distance": round(distance, 2),
                            "pickup_attempt": pickup_attempt,
                            "stuck_recovery": True,
                        }
                    self.ignored_pickups.add((round(current_objective[0]),
                                              round(current_objective[1]),
                                              current_objective[2]))
                    self._begin_backtrack(player)
                elif current_objective[2] in (AMMO_NAMES | HEALTH_NAMES
                                              | ARMOR_NAMES | POWERUP_NAMES):
                    self.ignored_pickups.add((round(current_objective[0]),
                                              round(current_objective[1]),
                                              current_objective[2]))
                    self._begin_backtrack(player)
                elif (current_objective[2] in (KEY_NAMES | {"exit"}) or
                      current_objective[2].startswith("switch:")):
                    # Repeated zero-progress recoveries are authoritative
                    # evidence that this directed approach is unusable, even
                    # when the agent cannot get close enough to start commit.
                    self.waypoint_stall_cycles[waypoint_key] += 1
                    self.stuck_events = 0
                    if (waypoint.target_sector >= 0 and
                            self.waypoint_stall_cycles[waypoint_key] >= 2):
                        self.blocked_edges.add((
                            current_sector, waypoint.target_sector,
                            float(waypoint.x), float(waypoint.y),
                        ))
                        self.route.clear()
                        self.route_source_sector = current_sector
                        self.backtrack_target = None
                        skip_stall_scan = True
                else:
                    self._begin_backtrack(player)
            if not skip_stall_scan:
                self.request_scan("navigation_stall")
            actions = {"turn left", "move backward", "jump"}
            if needs_use and not mission_portal:
                actions.add("use")
            self.last_positions.clear()
            self.distance_history.clear()
        if not actions:
            actions.add("move forward")
        return actions, {
            "mode": "navigate", "objective": current_objective[2],
            "objective_steps": self.objective_steps,
            "objective_x": current_objective[0], "objective_y": current_objective[1],
            "player_x": round(float(player.position_x), 2),
            "player_y": round(float(player.position_y), 2),
            "player_angle": round(float(player.angle), 2),
            "waypoint_x": waypoint.x, "waypoint_y": waypoint.y,
            "waypoint_special": waypoint.special, "waypoint_distance": round(distance, 2),
            "door_required_key": waypoint.required_key,
            "waypoints_remaining": len(self.route), "aim_error_degrees": round(error, 2),
            "stuck_recovery": stuck, "breadcrumbs": len(self.breadcrumbs),
            "blocked_portals": len(self.blocked_points),
            "mission_portal_retries": self.mission_portal_retries.get(waypoint_key, 0),
            "backtracking": self.backtrack_target is not None,
            "semantic_stall_hotspots": len(self.blocked_points),
            "current_sector": self.map.sector_at(player.position_x, player.position_y),
            "visited_sectors": len(self.sector_visits),
            "vertical_transition": vertical_transition,
            "floor_delta": waypoint.floor_delta,
            "red_light_jump_trigger": vertical_transition == "rise",
        }


def run_map(wad: Path, map_name: str, seed: int, visible: bool,
            timeout_seconds: float, retries: int, writer, trace_file,
            pace: bool = True, skill: int = 1,
            semantic_memory: dict | None = None) -> dict:
    truth = WadMap(wad, map_name)
    attempts = []
    total_video_tics = total_video_frames = 0
    for attempt in range(1, retries + 1):
        game = make_campaign_game(wad, map_name, seed + attempt, visible,
                                  timeout_seconds, skill)
        combat = TacticalOracle()
        navigation = CampaignNavigator(truth, semantic_memory)
        known_locked_doors = sorted({
            (float(portal.x), float(portal.y), portal.required_key)
            for portals in truth.graph.values() for portal in portals
            if portal.required_key
        })
        counts: Counter[str] = Counter()
        started = time.perf_counter()
        last_log = started
        game.new_episode()
        combat.reset(float(game.get_game_variable(vzd.GameVariable.HEALTH)))
        steps = 0
        combat_active = False
        previous_pickups: dict[tuple[str, int, int], tuple[float, float]] = {}
        try:
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break
                frame = np.ascontiguousarray(state.screen_buffer)
                combat_actions, combat_meta = combat.decide(state, game)
                sees_threat = has_valid_combat_threat(combat_meta)
                gate_objective = (navigation.objective[2]
                                  if navigation.objective is not None else None)
                progression_objective = bool(
                    gate_objective in (KEY_NAMES | {"exit"}) or
                    (gate_objective and gate_objective.startswith("switch:"))
                )
                gate_priority = bool(
                    (navigation.gate_commit_steps > 0 or
                     navigation.lift_transaction_steps > 0 or
                     navigation.detour_key is not None or
                     navigation.detour_transaction_target is not None) and
                    progression_objective
                )
                urgent_gate_combat = gate_combat_is_urgent(combat_meta)
                health = float(game.get_game_variable(vzd.GameVariable.HEALTH))
                reachable_health_visible = any(
                    item.name in HEALTH_NAMES for item in (state.objects or [])
                )
                critical_survival = health <= 30.0 and reachable_health_visible
                point_blank_threat = bool(
                    combat_meta.get("line_of_sight") and
                    float(combat_meta.get("target_distance", float("inf"))) <= 96.0 and
                    combat_meta.get("damaged")
                )
                if (sees_threat and (not gate_priority or urgent_gate_combat) and
                        (not critical_survival or point_blank_threat)):
                    actions, telemetry = combat_actions, combat_meta
                else:
                    # A key/exit gate is a transaction: once USE has been
                    # pulsed, preserve its route and remaining budget until the
                    # target sector is observed or the budget expires. Ordinary
                    # combat must not discard the crossing postcondition.
                    if gate_priority:
                        navigation.scan_steps = 0
                        navigation.scan_reason = None
                    elif (combat_active and gate_objective not in KEY_NAMES and
                          gate_objective != "exit"):
                        navigation.request_scan("room_cleared")
                    # Damage without a visible source means disengage and
                    # continue toward reachable health/key/exit objectives.
                    # Cancel long surveys and add one lateral evasive input;
                    # never let a hidden enemy hold the state machine in search.
                    evading_hidden_fire = bool(
                        combat_meta.get("under_fire") and
                        navigation.detour_transaction_target is None
                    )
                    if evading_hidden_fire:
                        navigation.scan_steps = 0
                        navigation.scan_reason = None
                    actions, telemetry = navigation.decide(
                        state,
                        float(game.get_game_variable(vzd.GameVariable.HEALTH)),
                        float(game.get_game_variable(vzd.GameVariable.ARMOR)),
                    )
                    if evading_hidden_fire:
                        actions.add("strafe left" if combat.strafe_sign < 0 else "strafe right")
                        telemetry["hidden_fire_evasion"] = True
                telemetry["gate_priority"] = gate_priority
                telemetry["urgent_gate_combat"] = urgent_gate_combat
                combat_active = sees_threat and (not gate_priority or urgent_gate_combat)
                for action in actions:
                    counts[action] += 1
                action_tics = int(telemetry.get("action_tics", TICS_PER_ACTION))
                action_vector = action_vector_for_names(actions, game)
                if telemetry.get("mode") == "precision_gap_crossing":
                    buttons = list(game.get_available_buttons())
                    delta_index = buttons.index(vzd.Button.MOVE_LEFT_RIGHT_DELTA)
                    action_vector[delta_index] = float(telemetry.get("lateral_delta", 0.0))
                reward = float(game.make_action(action_vector, action_tics))
                row = {
                    **provenance(),
                    "map": map_name, "attempt": attempt, "step": steps,
                    "tic": int(state.tic), "actions": sorted(actions), "reward": reward,
                    "health": float(game.get_game_variable(vzd.GameVariable.HEALTH)),
                    "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                    **telemetry,
                }
                if steps == 0:
                    row["map_locked_doors_known"] = [
                        {"x": x, "y": y, "required_key": required_key}
                        for x, y, required_key in known_locked_doors
                    ]
                telemetry_player = next(
                    item for item in state.objects if item.name == "DoomPlayer"
                )
                current_pickups = {
                    (item.name, round(item.position_x), round(item.position_y)):
                    (float(item.position_x), float(item.position_y))
                    for item in (state.objects or [])
                    if item.name in (WEAPON_NAMES | KEY_NAMES | HEALTH_NAMES)
                }
                row["pickups_confirmed"] = sorted({
                    name for (name, x, y), (px, py) in previous_pickups.items()
                    if (name, x, y) not in current_pickups and
                    math.hypot(telemetry_player.position_x - px,
                               telemetry_player.position_y - py) <= 72.0
                })
                previous_pickups = current_pickups
                row["weapon_pickups_visible"] = [
                    {
                        "name": item.name,
                        "x": float(item.position_x),
                        "y": float(item.position_y),
                        "distance": round(math.hypot(
                            item.position_x - telemetry_player.position_x,
                            item.position_y - telemetry_player.position_y,
                        ), 2),
                    }
                    for item in (state.objects or []) if item.name in WEAPON_NAMES
                ]
                row["key_pickups_visible"] = [
                    {
                        "name": item.name,
                        "x": float(item.position_x),
                        "y": float(item.position_y),
                        "distance": round(math.hypot(
                            item.position_x - telemetry_player.position_x,
                            item.position_y - telemetry_player.position_y,
                        ), 2),
                    }
                    for item in (state.objects or []) if item.name in KEY_NAMES
                ]
                row["health_pickups_visible"] = [
                    {
                        "name": item.name,
                        "x": float(item.position_x),
                        "y": float(item.position_y),
                        "distance": round(math.hypot(
                            item.position_x - telemetry_player.position_x,
                            item.position_y - telemetry_player.position_y,
                        ), 2),
                    }
                    for item in (state.objects or []) if item.name in HEALTH_NAMES
                ]
                if trace_file:
                    trace_file.write(json.dumps(row) + "\n")
                if time.perf_counter() - last_log >= 10:
                    print(json.dumps(row), flush=True)
                    last_log = time.perf_counter()
                total_video_tics += action_tics
                target_frames = round(total_video_tics * 30 / TICS_PER_SECOND)
                if writer:
                    marked_frame = stamp_frame(frame)
                    while total_video_frames < target_frames:
                        writer.append_data(marked_frame)
                        total_video_frames += 1
                steps += 1
                if pace:
                    time.sleep(action_tics / TICS_PER_SECOND)
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
    parser.add_argument("--maps", nargs="+")
    parser.add_argument("--episode", choices=("E1", "E2", "E3", "E4"),
                        help="Strict M1-M8 gauntlet; a failed mission blocks every later mission.")
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
    parser.add_argument("--semantic-memory", type=Path)
    args = parser.parse_args()
    if args.episode and args.maps:
        parser.error("choose either --episode or --maps, not both")
    requested_maps = episode_maps(args.episode) if args.episode else [
        name.upper() for name in (args.maps or ["E1M1"])
    ]
    wad = args.wad.resolve()
    semantic_memory = (json.loads(args.semantic_memory.read_text(encoding="utf-8"))
                       if args.semantic_memory else None)
    writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8) if args.output else None
    trace_file = args.trace.open("w", encoding="utf-8") if args.trace else None
    results = []
    try:
        for index, map_name in enumerate(requested_maps):
            result = run_map(wad, map_name.upper(), args.seed + index * 100,
                             args.visible, args.seconds_per_map, args.retries,
                             writer, trace_file, pace=not args.no_pace,
                             skill=args.skill, semantic_memory=semantic_memory)
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
        "wad": str(wad), "maps_requested": requested_maps,
        "maps_completed": sum(row["completed"] for row in results), "results": results,
        **gate_summary(requested_maps, results, args.episode),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
