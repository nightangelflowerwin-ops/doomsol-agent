from play_campaign_oracle import (
    critical_survival_combat_required,
    episode_maps,
    gate_combat_is_urgent,
    gate_summary,
    has_valid_combat_threat,
)


def test_episode_has_eight_mandatory_ordered_missions():
    assert episode_maps("e1") == [f"E1M{i}" for i in range(1, 9)]


def test_failed_mission_hard_blocks_later_missions():
    requested = episode_maps("E1")
    results = [
        {"map": "E1M1", "completed": True},
        {"map": "E1M2", "completed": False},
    ]
    summary = gate_summary(requested, results, "E1")
    assert summary["gate_status"] == "blocked"
    assert summary["blocked_at"] == "E1M2"
    assert summary["missions_attempted"] == ["E1M1", "E1M2"]
    assert not summary["episode_completed"]


def test_episode_completes_only_after_all_eight_missions():
    requested = episode_maps("E1")
    results = [{"map": name, "completed": True} for name in requested]
    summary = gate_summary(requested, results, "E1")
    assert summary["gate_status"] == "complete"
    assert summary["blocked_at"] is None
    assert summary["episode_completed"]


def test_hidden_damage_does_not_lock_combat_state():
    assert not has_valid_combat_threat({
        "line_of_sight": False,
        "visible_enemies": 0,
        "under_fire": True,
    })
    assert has_valid_combat_threat({
        "line_of_sight": True,
        "visible_enemies": 1,
        "under_fire": False,
    })


def test_gate_combat_requires_damage_or_close_visible_enemy():
    assert not gate_combat_is_urgent({
        "damaged": False, "line_of_sight": True, "target_distance": 300,
    })
    assert gate_combat_is_urgent({
        "damaged": False, "line_of_sight": True, "target_distance": 256,
    })
    assert gate_combat_is_urgent({
        "damaged": True, "line_of_sight": False, "target_distance": 900,
    })


def test_critical_survival_answers_only_visible_active_attackers():
    assert critical_survival_combat_required({
        "line_of_sight": True, "under_fire": True, "damaged": False,
    })
    assert critical_survival_combat_required({
        "line_of_sight": True, "under_fire": False, "damaged": True,
    })
    assert not critical_survival_combat_required({
        "line_of_sight": False, "under_fire": True, "damaged": True,
    })
    assert not critical_survival_combat_required({
        "line_of_sight": True, "under_fire": False, "damaged": False,
    })
