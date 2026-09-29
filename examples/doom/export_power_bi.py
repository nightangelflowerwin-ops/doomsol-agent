"""Export Doom campaign evidence as small relational tables for Power BI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STEP_FIELDS = (
    "run_id", "map", "attempt", "step", "tic", "mode", "objective",
    "health", "kills", "current_sector", "player_x", "player_y",
    "player_angle", "equipped_weapon_slot", "equipped_weapon_ammo",
    "target_name", "target_distance", "line_of_sight", "under_fire",
    "damaged", "reward", "actions",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_id_for(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


def export(run_dir: Path, output_dir: Path) -> dict[str, int]:
    attempts: list[dict] = []
    steps: list[dict] = []
    pickups: list[dict] = []
    transitions: list[dict] = []
    findings: list[dict] = []

    for summary_path in sorted(run_dir.glob("*-summary.json")):
        run_id = run_id_for(summary_path, "-summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for result in summary.get("results", []):
            for item in result.get("attempts", []):
                attempts.append({
                    "run_id": run_id,
                    "map": result.get("map"),
                    "attempt": item.get("attempt"),
                    "completed": item.get("completed"),
                    "dead": item.get("dead"),
                    "timed_out": item.get("timed_out"),
                    "steps": item.get("steps"),
                    "kills": item.get("kills"),
                    "items": item.get("items"),
                    "secrets": item.get("secrets"),
                    "wall_seconds": item.get("wall_seconds"),
                    "gate_status": summary.get("gate_status"),
                    "episode_completed": summary.get("episode_completed"),
                })

    for trace_path in sorted(run_dir.glob("*.jsonl")):
        run_id = run_id_for(trace_path, ".jsonl")
        previous_sector = None
        seen_pickup_opportunities: set[tuple[str, str, int, int]] = set()
        for row in read_jsonl(trace_path):
            exported = {field: row.get(field) for field in STEP_FIELDS}
            exported["run_id"] = run_id
            exported["actions"] = " | ".join(row.get("actions", []))
            steps.append(exported)

            sector = row.get("current_sector")
            if (previous_sector is not None and sector is not None and
                    sector != previous_sector):
                transitions.append({
                    "run_id": run_id, "step": row.get("step"),
                    "from_sector": previous_sector, "to_sector": sector,
                    "objective": row.get("objective"), "mode": row.get("mode"),
                })
            if sector is not None:
                previous_sector = sector

            weapon_visibility_category = (
                "visible_weapon" if "weapon_pickups_known" in row
                else "legacy_weapon_observation_unverified"
            )
            for category, field in (
                (weapon_visibility_category, "weapon_pickups_visible"),
                ("known_weapon", "weapon_pickups_known"),
                ("visible_key", "key_pickups_visible"),
                ("visible_health", "health_pickups_visible"),
            ):
                for pickup in row.get(field, []):
                    event = (
                        category, str(pickup.get("name")),
                        round(float(pickup.get("x", 0))),
                        round(float(pickup.get("y", 0))),
                    )
                    if event in seen_pickup_opportunities:
                        continue
                    seen_pickup_opportunities.add(event)
                    pickups.append({
                        "run_id": run_id, "step": row.get("step"),
                        "category": category, "name": pickup.get("name"),
                        "x": pickup.get("x"), "y": pickup.get("y"),
                        "distance": pickup.get("distance"), "confirmed": False,
                    })
            for name in row.get("pickups_confirmed", []):
                pickups.append({
                    "run_id": run_id, "step": row.get("step"),
                    "category": "confirmed", "name": name,
                    "x": None, "y": None, "distance": None, "confirmed": True,
                })

    for judge_path in sorted(run_dir.glob("*-judge.json")):
        run_id = run_id_for(judge_path, "-judge.json")
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        for failure in judge.get("failures", []):
            findings.append({
                "run_id": run_id, "judge": judge.get("judge"),
                "finding_type": "failure", "step": None, "detail": failure,
            })
        for field in (
            "pickup_regrets", "unsuitable_melee_events",
            "damage_response_failures", "wall_attack_streaks", "scan_failures",
            "no_progress_stalls",
        ):
            for finding in judge.get(field, []):
                findings.append({
                    "run_id": run_id, "judge": judge.get("judge"),
                    "finding_type": field, "step": finding.get("step"),
                    "detail": json.dumps(finding, sort_keys=True),
                })

    write_csv(output_dir / "attempts.csv", attempts, (
        "run_id", "map", "attempt", "completed", "dead", "timed_out",
        "steps", "kills", "items", "secrets", "wall_seconds", "gate_status",
        "episode_completed",
    ))
    write_csv(output_dir / "steps.csv", steps, STEP_FIELDS)
    write_csv(output_dir / "pickups.csv", pickups, (
        "run_id", "step", "category", "name", "x", "y", "distance", "confirmed",
    ))
    write_csv(output_dir / "route_transitions.csv", transitions, (
        "run_id", "step", "from_sector", "to_sector", "objective", "mode",
    ))
    write_csv(output_dir / "judge_findings.csv", findings, (
        "run_id", "judge", "finding_type", "step", "detail",
    ))
    return {
        "attempts": len(attempts), "steps": len(steps), "pickups": len(pickups),
        "route_transitions": len(transitions), "judge_findings": len(findings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True,
                        help="Folder containing trace, summary and judge artifacts")
    parser.add_argument("--output", type=Path, default=Path("power-bi-export"))
    args = parser.parse_args()
    print(json.dumps(export(args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
