"""Build persistent map memory and semantic-policy samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_memory import distill, write_distillation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wad", type=Path, required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    args = parser.parse_args()
    memory, samples = distill(args.wad.resolve(), args.map, args.trace)
    write_distillation(memory, samples, args.output, args.training_output)
    print(json.dumps({
        "map": memory["map"], "sectors": len(memory["sectors"]),
        "connections": len(memory["connections"]), "doors": len(memory["doors"]),
        "keys": len(memory["keys"]), "exits": len(memory["exits"]),
        "stall_hotspots": len(memory["stall_hotspots"]),
        "training_samples": len(samples), "output": str(args.output),
        "training_output": str(args.training_output),
    }, indent=2))


if __name__ == "__main__":
    main()
