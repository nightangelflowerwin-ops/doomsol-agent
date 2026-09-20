"""Deterministic hard-gate judge for recorded campaign attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WEAPONS = {"Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher",
           "PlasmaRifle", "BFG9000", "Chainsaw"}
KEYS = {"BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull"}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def route_evidence(rows: list[dict]) -> dict:
    sectors = [int(row["current_sector"]) for row in rows
               if row.get("current_sector") is not None]
    transitions = []
    for source, target in zip(sectors, sectors[1:]):
        edge = (source, target)
        if source != target and (not transitions or transitions[-1] != edge):
            transitions.append(edge)
    return {
        "sectors": set(sectors),
        "edges": set(transitions),
        "fingerprint": [f"{source}>{target}" for source, target in transitions],
    }


def cross_attempt_evidence(rows: list[dict], history: list[Path]) -> dict:
    current = route_evidence(rows)
    prior = []
    prior_sectors: set[int] = set()
    prior_edges: set[tuple[int, int]] = set()
    for path in history:
        evidence = route_evidence(load_rows(path))
        union = current["edges"] | evidence["edges"]
        similarity = (len(current["edges"] & evidence["edges"]) / len(union)
                      if union else 1.0)
        prior.append({
            "trace": path.name,
            "directed_edge_jaccard": round(similarity, 4),
        })
        prior_sectors |= evidence["sectors"]
        prior_edges |= evidence["edges"]
    closest = max(prior, key=lambda item: item["directed_edge_jaccard"],
                  default=None)
    return {
        "route_fingerprint": current["fingerprint"],
        "closest_prior_attempt": closest,
        "new_meaningful_sectors": sorted(current["sectors"] - prior_sectors),
        "new_directed_portals": [
            f"{source}>{target}" for source, target in
            sorted(current["edges"] - prior_edges)
        ],
    }


def judge(trace: Path, summary: Path,
          history: list[Path] | None = None) -> dict:
    rows = load_rows(trace)
    report = json.loads(summary.read_text(encoding="utf-8"))
    attempts = [attempt for result in report.get("results", [])
                for attempt in result.get("attempts", [])]
    genuine_completion = bool(
        report.get("episode_completed") and attempts and
        all(item.get("completed") and not item.get("dead") and
            not item.get("timed_out") for item in attempts)
    )

    # A scan spans many rows; judge its first row once, not every turn frame.
    room_clears = []
    previous_clear = False
    for row in rows:
        is_clear = row.get("scan_reason") == "room_cleared"
        if is_clear and not previous_clear:
            room_clears.append(row)
        previous_clear = is_clear
    room_exit_results = []
    for clear in room_clears:
        prior = [row.get("current_sector") for row in rows
                 if row.get("step", -1) <= clear.get("step", -1) and
                 row.get("current_sector") is not None]
        source = prior[-1] if prior else None
        window = [row for row in rows
                  if clear.get("step", -1) < row.get("step", -1)
                  <= clear.get("step", -1) + 44]
        transitioned = any(row.get("current_sector") is not None and
                           row.get("current_sector") != source for row in window)
        looted_weapon = any(
            set(row.get("pickups_confirmed", [])) & WEAPONS for row in window
        )
        combat_resumed = any(row.get("mode") == "engage" for row in window)
        room_exit_results.append({
            "step": clear.get("step"), "source_sector": source,
            "transitioned_within_5s": transitioned,
            "weapon_pickup_confirmed": looted_weapon,
            "combat_resumed": combat_resumed,
            # Seeing another enemy is not an exit. A cleared room passes only
            # when the agent changes sector or deliberately begins its nearby
            # confirmed weapon pickup within the bounded five-second window.
            "passed": transitioned or looted_weapon,
        })

    locked_door_rows = [row for row in rows if row.get("door_required_key")]
    mapped_locked_doors = [door for row in rows
                           for door in row.get("map_locked_doors_known", [])]
    weapon_objectives = [row for row in rows if row.get("objective") in WEAPONS]
    visible_weapon_rows = [row for row in rows if row.get("weapon_pickups_visible")]
    visible_key_rows = [row for row in rows if row.get("key_pickups_visible")]
    # Objective selection alone is not collection. Require the agent to get
    # close to the selected weapon and then observe it disappear from the
    # object list while remaining near the pickup coordinate.
    collected_weapons = []
    for index, row in enumerate(rows):
        if row.get("objective") not in WEAPONS:
            continue
        distance = ((float(row.get("player_x", 0)) - float(row.get("objective_x", 0))) ** 2 +
                    (float(row.get("player_y", 0)) - float(row.get("objective_y", 0))) ** 2) ** 0.5
        if distance > 56:
            continue
        name = row["objective"]
        later = rows[index + 1:index + 13]
        if any(name not in {item.get("name") for item in next_row.get(
                "weapon_pickups_visible", [])} for next_row in later):
            collected_weapons.append({"name": name, "step": row.get("step")})

    # A visible key is an immediate progression cue. It must become the
    # objective within two seconds, unless the agent is actively engaging an
    # immediate threat. Map knowledge alone is not a response to that cue.
    ignored_key_opportunities = []
    seen_key_events = set()
    for index, row in enumerate(rows):
        for item in row.get("key_pickups_visible", []):
            event = (item.get("name"), round(float(item.get("x", 0))),
                     round(float(item.get("y", 0))))
            if event in seen_key_events or float(item.get("distance", 1e9)) > 900:
                continue
            seen_key_events.add(event)
            window = rows[index:index + 19]
            responded = any(next_row.get("objective") == item.get("name") or
                            next_row.get("mode") == "engage" for next_row in window)
            if not responded:
                ignored_key_opportunities.append({
                    "name": item.get("name"), "step": row.get("step"),
                    "distance": item.get("distance"),
                })

    # A controller that spends more than roughly three seconds on the same
    # objective in the same sector without moving is failed immediately. This
    # catches wall pushing instead of allowing a five-minute proxy-score run.
    no_progress_stalls = []
    stall_window = 28
    for start in range(0, max(0, len(rows) - stall_window + 1)):
        window = rows[start:start + stall_window]
        first, last = window[0], window[-1]
        if (first.get("objective") != last.get("objective") or
                first.get("current_sector") != last.get("current_sector") or
                first.get("player_x") is None or last.get("player_x") is None):
            continue
        displacement = ((float(last["player_x"]) - float(first["player_x"])) ** 2 +
                        (float(last["player_y"]) - float(first["player_y"])) ** 2) ** 0.5
        if displacement < 12:
            no_progress_stalls.append({
                "start_step": first.get("step"), "end_step": last.get("step"),
                "objective": last.get("objective"),
                "sector": last.get("current_sector"),
                "displacement": round(displacement, 2),
            })
            # One incident is enough; overlapping windows add no information.
            break
    sectors = {row.get("current_sector") for row in rows
               if row.get("current_sector") is not None}
    history_evidence = cross_attempt_evidence(rows, history or [])
    closest = history_evidence["closest_prior_attempt"]
    repeated_route = bool(
        closest and closest["directed_edge_jaccard"] >= 0.85 and
        not history_evidence["new_meaningful_sectors"] and
        not history_evidence["new_directed_portals"]
    )
    failures = []
    if not genuine_completion:
        failures.append("No genuine sequential E1M1-E1M8 completion.")
    if room_exit_results and not all(item["passed"] for item in room_exit_results):
        failures.append("At least one cleared room was neither exited nor looted within five seconds.")
    if not locked_door_rows and not mapped_locked_doors:
        failures.append("No colored locked-door requirement was known or observed in telemetry.")
    if visible_weapon_rows and not weapon_objectives:
        failures.append("A dropped/crate weapon was visible but never selected as a pickup objective.")
    if weapon_objectives and not collected_weapons:
        failures.append("A weapon was selected but no pickup was confirmed.")
    if ignored_key_opportunities:
        failures.append("A visible key cue was ignored for more than two seconds without active combat.")
    if no_progress_stalls:
        failures.append("Hard no-progress stall: same sector/objective with under 12 units movement.")
    if repeated_route:
        failures.append(
            "cross_attempt_repetition: failed route repeated with at least "
            "0.85 directed-edge similarity and no new map frontier."
        )

    return {
        "judge": "movingman_route_memory_judge_v2",
        "trace": str(trace.resolve()),
        "completion_reward": 1 if genuine_completion else 0,
        "genuine_completion": genuine_completion,
        "maps_completed": report.get("maps_completed", 0),
        "room_clear_checks": room_exit_results,
        "locked_door_sign_observations": len(locked_door_rows),
        "map_locked_door_requirements": mapped_locked_doors,
        "weapon_pickup_objective_steps": sorted({row.get("step") for row in weapon_objectives}),
        "confirmed_weapon_pickups": collected_weapons,
        "visible_weapon_opportunity_steps": sorted({row.get("step") for row in visible_weapon_rows}),
        "visible_key_opportunity_steps": sorted({row.get("step") for row in visible_key_rows}),
        "ignored_key_opportunities": ignored_key_opportunities,
        "no_progress_stalls": no_progress_stalls,
        "sectors_observed": sorted(sectors),
        "cross_attempt": history_evidence,
        "cross_attempt_repetition": repeated_route,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history", type=Path, action="append", default=[])
    args = parser.parse_args()
    current = args.trace.resolve()
    history = [path for path in args.history if path.resolve() != current]
    result = judge(args.trace, args.summary, history=history)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
