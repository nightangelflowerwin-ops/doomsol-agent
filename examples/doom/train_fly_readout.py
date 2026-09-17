"""Time-bounded teacher imitation for either experimental fly readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, make_game
from fly_navigation import FlyInspiredNavigation, FullConnectomeReservoir
from jevlike.provenance import provenance
from jevlike.vision import observation, observation_tensor
from train_imitation import expert_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("compact", "connectome"), required=True)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    game = make_game(args.seed, "deadly_corridor", teacher_truth=True)
    compact = FlyInspiredNavigation(actions=len(DEFEND_ACTIONS)) if args.controller == "compact" else None
    reservoir = FullConnectomeReservoir(actions=len(DEFEND_ACTIONS), seed=args.seed) if compact is None else None
    parameters = compact.policy.parameters() if compact is not None else reservoir.readout.parameters()
    optimiser = torch.optim.AdamW(parameters, lr=2e-3, weight_decay=1e-4)
    features, labels = [], []
    state = previous = None
    episodes = updates = examples = 0
    kills, rewards = [], []
    losses, accuracies = [], []
    action_counts = np.zeros(len(DEFEND_ACTIONS), np.int64)
    started = time.perf_counter()
    game.new_episode()
    try:
        while time.perf_counter() - started < args.seconds:
            if game.is_episode_finished():
                kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
                rewards.append(float(game.get_total_reward()))
                episodes += 1
                game.new_episode(); previous = state = None
                if reservoir is not None:
                    reservoir.reset(args.seed + episodes)
            current = game.get_state()
            if current is None:
                continue
            frame = np.ascontiguousarray(current.screen_buffer)
            label = expert_action(current, game=game)
            action_counts[label] += 1
            if compact is not None:
                item = observation(frame, previous)
                with torch.no_grad():
                    sectors, context = compact.sensory_features(observation_tensor([item], torch.device("cpu")))
                    if state is None:
                        state = compact.initial_state(1)
                    sensory = compact.visual_projection(sectors) + compact.context(context)
                    state = torch.tanh(state @ compact.ring_recurrence.T + sensory).detach()
                    feature = torch.cat((state, context), 1)[0]
            else:
                _, array = reservoir.step(frame)
                feature = torch.from_numpy(array)
            features.append(feature)
            labels.append(label)
            examples += 1
            game.make_action(action_vector_for_game(label, DEFEND_ACTIONS, game), TICS_PER_ACTION)
            previous = frame
            if len(features) >= args.batch_size:
                x = torch.stack(features); y = torch.tensor(labels)
                head = compact.policy if compact is not None else reservoir.readout
                logits = head(x)
                counts = torch.bincount(y, minlength=len(DEFEND_ACTIONS)).float()
                weights = torch.ones_like(counts)
                present = counts > 0
                weights[present] = (len(y) / counts[present]).sqrt().clamp(max=3.0)
                loss = F.cross_entropy(logits, y, weight=weights)
                optimiser.zero_grad(); loss.backward(); optimiser.step()
                losses.append(float(loss.detach()))
                accuracies.append(float((logits.argmax(1) == y).float().mean()))
                updates += 1; features.clear(); labels.clear()
            if examples % 512 == 0:
                print(json.dumps({"controller": args.controller, "elapsed": round(time.perf_counter()-started, 1),
                                  "examples": examples, "updates": updates,
                                  "loss": round(losses[-1], 4) if losses else None,
                                  "accuracy": round(accuracies[-1], 4) if accuracies else None}), flush=True)
    finally:
        game.close()
    payload = {
        **provenance(), "version": 1, "controller": args.controller,
        "model": compact.state_dict() if compact is not None else reservoir.readout.state_dict(),
        "actions": DEFEND_ACTIONS, "scenario": "deadly_corridor", "seed": args.seed,
        "training_wall_seconds": round(time.perf_counter() - started, 3),
        "examples": examples, "updates": updates, "episodes": episodes,
        "mean_kills": float(np.mean(kills)) if kills else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "action_counts": action_counts.tolist(),
        "final_loss": losses[-1] if losses else None,
        "final_batch_accuracy": accuracies[-1] if accuracies else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({k:v for k,v in payload.items() if k != "model"}, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k != "model"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
