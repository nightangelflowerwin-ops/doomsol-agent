from play_campaign_oracle import episode_maps, gate_summary


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
