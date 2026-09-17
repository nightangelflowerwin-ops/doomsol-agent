"""Persistent semantic map memory distilled from WAD truth and replay traces."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jevlike.provenance import provenance
from wad_navigation import DOOR_SPECIALS, WadMap


def distill(wad: Path, map_name: str, traces: list[Path]) -> tuple[dict, list[dict]]:
    truth = WadMap(wad, map_name)
    edges = []
    seen = set()
    for source, portals in truth.graph.items():
        for portal in portals:
            key = tuple(sorted((source, portal.target_sector))) + (portal.x, portal.y)
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "source_sector": source, "target_sector": portal.target_sector,
                "x": portal.x, "y": portal.y, "special": portal.special,
                "floor_delta": portal.floor_delta, "opening": portal.opening,
                "kind": ("door" if portal.special in DOOR_SPECIALS else
                         "lift_or_jump" if portal.floor_delta > 24 else
                         "drop" if portal.floor_delta < -24 else "passage"),
            })
    stalls: Counter[tuple[float, float]] = Counter()
    samples = []
    for trace in traces:
        if not trace.exists():
            continue
        with trace.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("map", map_name).upper() != map_name.upper():
                    continue
                x, y = row.get("waypoint_x"), row.get("waypoint_y")
                if x is not None and y is not None and (
                    row.get("stuck_recovery") or row.get("scan_reason") == "navigation_stall"
                ):
                    stalls[(round(float(x), 1), round(float(y), 1))] += 1
                samples.append({
                    "map": map_name.upper(), "tic": row.get("tic"),
                    "semantic_mode": row.get("mode"), "objective": row.get("objective"),
                    "waypoint": [x, y] if x is not None and y is not None else None,
                    "door_ahead": row.get("waypoint_special") in DOOR_SPECIALS,
                    "vertical_transition": row.get("vertical_transition", "level"),
                    "red_light_jump_trigger": bool(row.get("red_light_jump_trigger", False)),
                    "exit_goal": row.get("objective") == "exit",
                    "stalled": bool(row.get("stuck_recovery")
                                    or row.get("scan_reason") == "navigation_stall"),
                    "actions": row.get("actions", []),
                })
    memory = {
        **provenance(), "schema_version": 1, "map": map_name.upper(),
        "sectors": sorted(truth.sector_lines), "connections": edges,
        "doors": [edge for edge in edges if edge["kind"] == "door"],
        "keys": [{"x": x, "y": y, "kind": kind} for x, y, kind in truth.key_points],
        "exits": [{"x": x, "y": y, "special": special}
                  for x, y, special in truth.exit_points],
        "stall_hotspots": [
            {"x": x, "y": y, "observations": count}
            for (x, y), count in stalls.most_common()
        ],
        "source_traces": [str(path.resolve()) for path in traces],
        "training_samples": len(samples),
    }
    return memory, samples


def write_distillation(memory: dict, samples: list[dict], output: Path,
                       training_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    training_output.parent.mkdir(parents=True, exist_ok=True)
    with training_output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
