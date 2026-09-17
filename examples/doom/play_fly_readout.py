"""Record a bounded evaluation video for a trained fly-controller readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import vizdoom as vzd
from PIL import Image

from environment import TICS_PER_ACTION, TICS_PER_SECOND, action_vector_for_game, make_game
from fly_navigation import FlyInspiredNavigation, FullConnectomeReservoir
from jevlike.vision import observation, observation_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--game-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actions = tuple(payload["actions"])
    compact = FlyInspiredNavigation(actions=len(actions)) if payload["controller"] == "compact" else None
    reservoir = FullConnectomeReservoir(actions=len(actions), seed=payload["seed"] + 1000) if compact is None else None
    (compact if compact is not None else reservoir.readout).load_state_dict(payload["model"])
    game = make_game(payload["seed"] + 1000, payload["scenario"], "640x480")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
    state = previous = None
    video_frames = video_tics = steps = episode = 0
    kills, rewards = [], []
    limit = round(args.game_seconds * TICS_PER_SECOND / TICS_PER_ACTION)
    game.new_episode(); episode += 1
    try:
        while steps < limit:
            if game.is_episode_finished():
                kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
                rewards.append(float(game.get_total_reward()))
                game.new_episode(); episode += 1; state = previous = None
                if reservoir is not None: reservoir.reset(payload["seed"] + 1000 + episode)
            current = game.get_state()
            if current is None: continue
            frame = np.ascontiguousarray(current.screen_buffer)
            policy_frame = np.asarray(Image.fromarray(frame).resize((160, 120)))
            with torch.inference_mode():
                if compact is not None:
                    logits, _, state = compact(observation_tensor([observation(policy_frame, previous)], torch.device("cpu")), state)
                else:
                    logits, _ = reservoir.step(policy_frame)
            action = int(logits.argmax(1))
            game.make_action(action_vector_for_game(action, actions, game), TICS_PER_ACTION)
            video_tics += TICS_PER_ACTION
            target = round(video_tics * 30 / TICS_PER_SECOND)
            while video_frames < target:
                writer.append_data(frame); video_frames += 1
            previous = policy_frame; steps += 1
    finally:
        if not game.is_episode_finished():
            kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
            rewards.append(float(game.get_total_reward()))
        writer.close(); game.close()
    report = {"controller": payload["controller"], "game_seconds": args.game_seconds,
              "episodes": episode, "mean_kills": float(np.mean(kills)),
              "mean_reward": float(np.mean(rewards)), "video": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
