"""Bounded offline smoke benchmark for both fly-controller experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from fly_navigation import FlyInspiredNavigation, FullConnectomeReservoir
from jevlike.provenance import provenance


def frames(path: Path | None, count: int) -> list[np.ndarray]:
    result = []
    if path:
        for index, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
            if index % 8 == 0:
                result.append(np.asarray(Image.fromarray(frame[..., :3]).resize((160, 120))))
            if len(result) >= count:
                break
    if not result:
        rng = np.random.default_rng(17)
        result = [rng.integers(0, 256, (120, 160, 3), np.uint8) for _ in range(count)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sample = frames(args.video, args.frames)

    compact = FlyInspiredNavigation().eval()
    state = None
    compact_times, compact_actions = [], []
    previous = None
    for frame in sample:
        motion = np.zeros(frame.shape[:2], np.uint8) if previous is None else np.clip(
            (frame.astype(np.float32) - previous.astype(np.float32)).mean(-1) + 128, 0, 255
        ).astype(np.uint8)
        item = np.concatenate((frame, motion[..., None]), -1)
        tensor = torch.from_numpy(item).permute(2, 0, 1).float().div(255).unsqueeze(0)
        started = time.perf_counter()
        with torch.inference_mode():
            logits, _, state = compact(tensor, state)
        compact_times.append((time.perf_counter() - started) * 1000)
        compact_actions.append(int(logits.argmax(1)))
        previous = frame

    reservoir = FullConnectomeReservoir()
    cold_started = time.perf_counter()
    reservoir.step(sample[0])
    cold_start_ms = (time.perf_counter() - cold_started) * 1000
    reservoir.reset()
    reservoir_times, reservoir_actions, active_bins = [], [], []
    for frame in sample:
        started = time.perf_counter()
        with torch.inference_mode():
            logits, features = reservoir.step(frame)
        reservoir_times.append((time.perf_counter() - started) * 1000)
        reservoir_actions.append(int(logits.argmax(1)))
        active_bins.append(int(np.count_nonzero(features)))

    report = {
        **provenance(),
        "status": "offline_smoke_only_not_trained_or_promoted",
        "frames": len(sample),
        "compact_fly_inspired": {
            "mean_ms": round(float(np.mean(compact_times)), 3),
            "p95_ms": round(float(np.percentile(compact_times, 95)), 3),
            "unique_actions": len(set(compact_actions)),
        },
        "full_malecns_reservoir": {
            "neurons": reservoir.neuron_count,
            "descending_neurons": len(reservoir.descending),
            "cold_start_jit_ms": round(cold_start_ms, 3),
            "mean_ms": round(float(np.mean(reservoir_times)), 3),
            "p95_ms": round(float(np.percentile(reservoir_times, 95)), 3),
            "mean_active_readout_bins": round(float(np.mean(active_bins)), 3),
            "unique_actions": len(set(reservoir_actions)),
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
