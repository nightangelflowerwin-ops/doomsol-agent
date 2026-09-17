import json
from pathlib import Path

from semantic_memory import distill


WAD = Path(__file__).resolve().parents[3] / "dwasm-sim" / "wasm" / "fs" / "freedoom1.wad"


def test_distillation_contains_static_semantics_and_stall_labels(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "map": "E1M1", "tic": 4, "mode": "survey_360",
        "scan_reason": "navigation_stall", "waypoint_x": 10, "waypoint_y": 20,
        "waypoint_special": 1, "objective": "BlueCard", "actions": ["turn left"],
    }) + "\n", encoding="utf-8")
    memory, samples = distill(WAD, "E1M1", [trace])
    assert memory["doors"] and memory["exits"] and memory["keys"]
    assert memory["stall_hotspots"][0]["observations"] == 1
    assert samples[0]["door_ahead"] and samples[0]["stalled"]
