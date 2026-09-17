"""Small checks for the shared visual scorer and seven-button controller."""

import numpy as np
import torch
from types import SimpleNamespace

from diagnostics import enemy_observation
from environment import DOOM_ACTIONS, DOOM_ACTIONS_WITH_USE, action_vector_for_game, make_game
from train_imitation import expert_action
from jevlike.vision import (
    CHESS_OPTION_IDS,
    TOTAL_OPTIONS,
    DoomScorerV2,
    observation,
    observation_tensor,
)


def test_visual_scorer_preserves_each_option_axis() -> None:
    model = DoomScorerV2(actions=TOTAL_OPTIONS)
    current = np.full((120, 160, 3), 90, dtype=np.uint8)
    previous = np.full((120, 160, 3), 80, dtype=np.uint8)
    item = observation(current, previous)
    logits, value, trace = model.forward_trace(
        observation_tensor([item], torch.device("cpu")), torch.arange(7)
    )
    assert item.shape == (120, 160, 4)
    assert logits.shape == (1, len(DOOM_ACTIONS))
    assert value.shape == (1,)
    assert trace["attention_map"].shape == (1, len(DOOM_ACTIONS), 80)
    assert torch.allclose(trace["probabilities"].sum(-1), torch.ones(1))


def test_one_table_selects_the_five_chess_keys() -> None:
    model = DoomScorerV2(actions=TOTAL_OPTIONS)
    item = np.zeros((120, 160, 4), dtype=np.uint8)
    item[..., 3] = 128
    logits, value = model(
        observation_tensor([item], torch.device("cpu")),
        torch.tensor(CHESS_OPTION_IDS),
    )
    assert model.options.num_embeddings == 12
    assert logits.shape == (1, 5)
    assert value.shape == (1,)


def test_use_action_is_available_without_changing_legacy_controller() -> None:
    game = make_game(seed=11, include_use=True)
    try:
        assert len(DOOM_ACTIONS) == 7
        assert DOOM_ACTIONS_WITH_USE[-1] == "use"
        use = action_vector_for_game(
            DOOM_ACTIONS_WITH_USE.index("use"), DOOM_ACTIONS_WITH_USE, game
        )
        assert sum(use) == 1
        assert use[-1] is True
        game.make_action(use, 1)
    finally:
        game.close()


def test_enemy_teacher_uses_relative_position_and_visibility() -> None:
    enemy = SimpleNamespace(
        object_name="DoomImp", x=65, width=20, height=40,
        object_position_x=100.0, object_position_y=0.0, object_angle=180.0,
    )
    state = SimpleNamespace(
        labels=[enemy], screen_buffer=np.zeros((120, 160, 3), dtype=np.uint8)
    )

    class FakeGame:
        def get_game_variable(self, variable):
            values = {"POSITION_X": 0.0, "POSITION_Y": 0.0}
            return values[variable.name]

    observed = enemy_observation(state, FakeGame())
    assert observed is not None
    assert observed["mutual_line_of_sight"] is True
    assert observed["facing_agent"] is True
    assert observed["distance"] == 100.0
    assert abs(observed["x"] - -0.0625) < 1e-6


def test_aligned_enemy_teacher_interleaves_fire_and_evasion() -> None:
    enemy = SimpleNamespace(
        object_name="DoomImp", x=70, width=20, height=40,
        object_position_x=100.0, object_position_y=0.0, object_angle=180.0,
    )
    state = SimpleNamespace(
        labels=[enemy], screen_buffer=np.zeros((120, 160, 3), dtype=np.uint8)
    )

    class FakeGame:
        episode_time = 0

        def get_game_variable(self, variable):
            return 0.0

        def get_episode_time(self):
            return self.episode_time

    game = FakeGame()
    actions = []
    for phase in range(3):
        game.episode_time = phase * 4
        actions.append(DOOM_ACTIONS[expert_action(state, game=game)])
    assert actions == ["attack", "strafe left", "strafe right"]
