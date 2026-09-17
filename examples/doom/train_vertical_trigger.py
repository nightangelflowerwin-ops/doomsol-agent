"""Train a pixel classifier for level, rise/jump and drop transitions."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jevlike.provenance import provenance
from wad_navigation import WadMap


LABELS = ("level", "rise_jump", "drop")


def validation_metrics(confusion: np.ndarray) -> dict:
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        actual = int(confusion[index, :].sum())
        per_class[label] = {
            "support": actual,
            "precision": round(true_positive / predicted, 4) if predicted else None,
            "recall": round(true_positive / actual, 4) if actual else None,
        }
    rise = per_class["rise_jump"]
    drop = per_class["drop"]
    promoted = bool(
        rise["support"] >= 30
        and (rise["precision"] or 0) >= 0.70
        and (rise["recall"] or 0) >= 0.70
        and drop["support"] >= 20
        and (drop["precision"] or 0) >= 0.70
        and (drop["recall"] or 0) >= 0.70
    )
    return {
        "per_class": per_class,
        "promotion": {
            "approved_for_live_control": promoted,
            "reason": "validation thresholds passed" if promoted else
                "insufficient safe validation; checkpoint remains observation-only",
        },
    }


class VerticalPixels(Dataset):
    def __init__(self, frames: list[np.ndarray], labels: list[int]) -> None:
        self.frames = frames
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        frame = Image.fromarray(self.frames[index]).resize((96, 72))
        tensor = torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1).float() / 255.0
        return tensor, self.labels[index]


class VerticalTriggerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
        )
        self.head = nn.Linear(48 * 3 * 4, len(LABELS))

    def forward(self, value):
        return self.head(self.features(value))


def label_rows(trace: Path, truth: WadMap) -> list[dict]:
    portals = [portal for values in truth.graph.values() for portal in values]
    rows = []
    with trace.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("waypoint_x") is None or row.get("waypoint_y") is None:
                continue
            nearest = min(portals, key=lambda portal: math.hypot(
                portal.x - float(row["waypoint_x"]), portal.y - float(row["waypoint_y"])
            ))
            distance = math.hypot(nearest.x - float(row["waypoint_x"]),
                                  nearest.y - float(row["waypoint_y"]))
            delta = nearest.floor_delta if distance <= 64 else 0
            label = 1 if delta > 16 else 2 if delta < -24 else 0
            rows.append({"step": int(row["step"]), "label": label,
                         "floor_delta": delta, "portal_distance": distance})
    return rows


def load_frames(video: Path, rows: list[dict]) -> tuple[list[np.ndarray], list[int]]:
    by_frame = {
        round(row["step"] * 30 * 4 / 35): row["label"] for row in rows
    }
    frames, labels = [], []
    for index, frame in enumerate(iio.imiter(video, plugin="FFMPEG")):
        if index in by_frame:
            frames.append(np.asarray(frame)[..., :3])
            labels.append(by_frame[index])
        if index > max(by_frame, default=-1):
            break
    return frames, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--wad", type=Path, required=True)
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    rows = label_rows(args.trace, WadMap(args.wad.resolve(), args.map))
    frames, labels = load_frames(args.video, rows)
    if len(set(labels)) < 2:
        raise RuntimeError(f"need at least two vertical classes, got {dict(zip(LABELS, np.bincount(labels, minlength=3)))}")
    split = max(1, int(len(labels) * 0.8))
    train = VerticalPixels(frames[:split], labels[:split])
    valid = VerticalPixels(frames[split:], labels[split:])
    counts = np.bincount(labels[:split], minlength=3)
    weights = torch.tensor([len(train) / max(1, count) for count in counts], dtype=torch.float32)
    model = VerticalTriggerNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in DataLoader(train, batch_size=64, shuffle=True):
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); correct = total = 0; confusion = np.zeros((3, 3), dtype=int)
        with torch.no_grad():
            for x, y in DataLoader(valid, batch_size=64):
                prediction = model(x).argmax(1)
                correct += int((prediction == y).sum()); total += len(y)
                for actual, predicted in zip(y.tolist(), prediction.tolist()):
                    confusion[actual, predicted] += 1
        history.append({"epoch": epoch, "loss": round(sum(losses) / len(losses), 5),
                        "validation_accuracy": round(correct / max(1, total), 4)})
        print(json.dumps(history[-1]), flush=True)
    payload = {
        **provenance(), "model": "vertical_trigger_pixels_v1", "labels": LABELS,
        "state_dict": model.state_dict(), "source_video": str(args.video.resolve()),
        "source_trace": str(args.trace.resolve()), "class_counts": np.bincount(labels, minlength=3).tolist(),
        "train_samples": len(train), "validation_samples": len(valid),
        "history": history, "confusion": confusion.tolist(),
        "validation": validation_metrics(confusion),
        "training_seconds": round(time.time() - started, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {key: value for key, value in payload.items() if key not in {"state_dict"}}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
