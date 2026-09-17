"""Experimental fly-inspired and full-connectome Doom controllers.

Neither controller is promoted to live play by this module.  The compact model
is trainable end-to-end.  The reservoir keeps the MaleCNS wiring frozen and
only exposes a small trainable action readout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FlyInspiredNavigation(nn.Module):
    """Small ring-attractor navigation policy driven by pixels and optic flow."""

    def __init__(self, actions: int = 7, headings: int = 16) -> None:
        super().__init__()
        self.actions = actions
        self.headings = headings
        self.visual_projection = nn.Linear(headings * 2, headings)
        self.context = nn.Sequential(nn.Linear(4, headings), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(headings + 4, 64), nn.SiLU(),
                                    nn.Linear(64, actions))
        self.value = nn.Sequential(nn.Linear(headings + 4, 32), nn.SiLU(), nn.Linear(32, 1))
        recurrent = torch.zeros(headings, headings)
        recurrent.fill_diagonal_(0.55)
        recurrent += torch.roll(torch.eye(headings), 1, 0) * 0.20
        recurrent += torch.roll(torch.eye(headings), -1, 0) * 0.20
        self.register_buffer("ring_recurrence", recurrent)

    def initial_state(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch, self.headings, device=device)

    def sensory_features(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 4 or observations.shape[1] != 4:
            raise ValueError("observations must have shape (batch, 4, height, width)")
        rgb = observations[:, :3].mean(1, keepdim=True)
        motion = observations[:, 3:4] * 2 - 1
        sectors = torch.cat((
            F.adaptive_avg_pool2d(rgb, (1, self.headings)).flatten(1),
            F.adaptive_avg_pool2d(motion.abs(), (1, self.headings)).flatten(1),
        ), 1)
        left = motion[..., : motion.shape[-1] // 2].abs().mean((1, 2, 3))
        right = motion[..., motion.shape[-1] // 2 :].abs().mean((1, 2, 3))
        forward = motion.abs().mean((1, 2, 3))
        darkness = 1 - rgb.mean((1, 2, 3))
        context = torch.stack((left, right, forward, darkness), 1)
        return sectors, context

    def forward(self, observations: torch.Tensor, state: torch.Tensor | None = None):
        sectors, context = self.sensory_features(observations)
        if state is None:
            state = self.initial_state(len(observations), observations.device)
        sensory = self.visual_projection(sectors) + self.context(context)
        state = torch.tanh(state @ self.ring_recurrence.T + sensory)
        combined = torch.cat((state, context), 1)
        return self.policy(combined), self.value(combined).squeeze(-1), state


class FullConnectomeReservoir:
    """Frozen 166,700-neuron MaleCNS reservoir with a compact Doom readout."""

    def __init__(self, actions: int = 7, bins: int = 64, seed: int = 17) -> None:
        try:
            from flybrain import FlyBrain
        except ImportError as error:
            raise RuntimeError("install the optional 'fly' dependencies") from error
        self.brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
        self.actions = actions
        self.bins = bins
        self.descending = self.brain.cells(["descending_neuron"])
        if not len(self.descending):
            raise RuntimeError("MaleCNS metadata contains no descending neurons")
        self.descending_lookup = np.full(self.brain.n, -1, np.int32)
        self.descending_lookup[self.descending] = np.arange(len(self.descending), dtype=np.int32)
        self.readout = nn.Linear(bins, actions)

    @property
    def neuron_count(self) -> int:
        return self.brain.n

    def reset(self, seed: int = 17) -> None:
        self.brain.reset(seed)

    def eye_drive(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape[0] == 4:
            observation = np.moveaxis(observation, 0, -1)
        rgb = observation[..., :3].astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
        horizontal = rgb.mean((0, 2))
        columns = np.clip(np.rint((self.brain.azimuth + 1) * 0.5 * (len(horizontal) - 1)),
                          0, len(horizontal) - 1).astype(np.int32)
        return np.clip(horizontal[columns], 0, 1).astype(np.float32)

    def step(self, observation: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        fired = self.brain.step(eye_drive=self.eye_drive(observation))
        descending_positions = self.descending_lookup[fired]
        descending_positions = descending_positions[descending_positions >= 0]
        features = np.bincount(descending_positions % self.bins, minlength=self.bins).astype(np.float32)
        features /= max(1.0, features.sum())
        tensor = torch.from_numpy(features).unsqueeze(0)
        return self.readout(tensor), features

