"""Admission checks for Dwasm teacher trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest_frame(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="ignore")).hexdigest()


def validate(rows: list[dict], expected_seconds: int | None, require_use: bool) -> dict:
    ticks = [int(row["state"]["tic"]) for row in rows]
    positions = [(round(float(row["state"]["x"]), 3), round(float(row["state"]["y"]), 3)) for row in rows]
    frames = {digest_frame(str(row.get("frame", ""))) for row in rows}
    health = [int(row["state"]["health"]) for row in rows]
    kills = [int(row["state"]["kills"]) for row in rows]
    actions = [row["action"] for row in rows]
    capture_every = int(rows[0]["state"].get("teacher", {}).get("capture_every", 0)) if rows else 0
    expected_rows = None
    if expected_seconds is not None and capture_every:
        expected_rows = expected_seconds * 35 // capture_every

    failures: list[str] = []
    if len(rows) < 2:
        failures.append("fewer than two samples")
    if expected_rows is not None and len(rows) != expected_rows:
        failures.append(f"expected {expected_rows} samples, got {len(rows)}")
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        failures.append("ticks are not strictly increasing")
    if len(set(positions)) < max(2, len(rows) // 10):
        failures.append("insufficient position change")
    if len(frames) < max(2, len(rows) // 10):
        failures.append("insufficient visual change")
    if not any(abs(float(a.get("forward", 0))) > 0 or abs(float(a.get("strafe", 0))) > 0 for a in actions):
        failures.append("no movement actions")
    if not any(bool(a.get("fire")) for a in actions):
        failures.append("no fire actions")
    if require_use and not any(bool(a.get("use")) for a in actions):
        failures.append("no door/use actions")
    if max(kills, default=0) <= min(kills, default=0):
        failures.append("no kill progression")
    if len(set(health)) < 2:
        failures.append("no health change")

    return {
        "admitted": not failures,
        "samples": len(rows),
        "expected_samples": expected_rows,
        "first_tick": ticks[0] if ticks else None,
        "last_tick": ticks[-1] if ticks else None,
        "unique_positions": len(set(positions)),
        "unique_frames": len(frames),
        "fire_samples": sum(bool(a.get("fire")) for a in actions),
        "use_samples": sum(bool(a.get("use")) for a in actions),
        "health_range": [min(health), max(health)] if health else None,
        "kill_range": [min(kills), max(kills)] if kills else None,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-seconds", type=int)
    parser.add_argument("--require-use", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare", type=Path,
                        help="Require exact action/state equality with another manifest")
    args = parser.parse_args()
    report = validate(load(args.manifest), args.expected_seconds, args.require_use)
    if args.compare:
        left = load(args.manifest)
        right = load(args.compare)
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            if a.get("tick") != b.get("tick") or a.get("action") != b.get("action") or a.get("state") != b.get("state"):
                if len(differences) < 20:
                    differences.append(index)
        difference_count = sum(
            a.get("tick") != b.get("tick") or a.get("action") != b.get("action") or a.get("state") != b.get("state")
            for a, b in zip(left, right)
        ) + abs(len(left) - len(right))
        report["determinism"] = {
            "reference": str(args.compare),
            "sample_count_match": len(left) == len(right),
            "authoritative_difference_count": difference_count,
            "first_difference_indices": differences,
            "frame_bytes_compared": False,
            "note": "JPEG bytes are excluded; ticks, actions and authoritative state must match exactly.",
        }
        if difference_count:
            report["admitted"] = False
            report["failures"].append(f"{difference_count} deterministic action/state differences")
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    sys.exit(main())
