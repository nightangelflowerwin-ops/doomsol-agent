import csv
import json

from export_power_bi import export


def test_export_builds_relational_power_bi_tables(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "sample-summary.json").write_text(json.dumps({
        "gate_status": "blocked", "episode_completed": False,
        "results": [{"map": "E1M2", "attempts": [{
            "attempt": 1, "completed": False, "dead": True,
            "timed_out": False, "steps": 2, "kills": 1, "items": 0,
            "secrets": 0, "wall_seconds": 1.5,
        }]}],
    }), encoding="utf-8")
    trace = [
        {"step": 0, "map": "E1M2", "current_sector": 3,
         "actions": ["move forward"], "weapon_pickups_visible": [{
             "name": "Shotgun", "x": 1, "y": 2, "distance": 30,
         }]},
        {"step": 1, "map": "E1M2", "current_sector": 4,
         "actions": ["attack"], "pickups_confirmed": ["Shotgun"]},
    ]
    (runs / "sample.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace), encoding="utf-8"
    )
    (runs / "sample-judge.json").write_text(json.dumps({
        "judge": "test", "failures": ["No exit"],
    }), encoding="utf-8")

    counts = export(runs, tmp_path / "out")
    assert counts == {"attempts": 1, "steps": 2, "pickups": 2,
                      "route_transitions": 1, "judge_findings": 1}
    with (tmp_path / "out" / "attempts.csv").open(encoding="utf-8-sig") as handle:
        assert list(csv.DictReader(handle))[0]["run_id"] == "sample"
