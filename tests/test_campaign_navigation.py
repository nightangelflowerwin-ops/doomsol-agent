from pathlib import Path
from types import SimpleNamespace
import sys


DOOM_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "doom"
sys.path.insert(0, str(DOOM_EXAMPLES))

from play_campaign_oracle import (  # noqa: E402
    CampaignNavigator,
    PORTAL_STAGE_SPEED_TOLERANCE,
)
from wad_navigation import Portal  # noqa: E402


class TaggedKeyMap:
    switch_points = [
        (1664.0, -992.0, 23, 3, 1664.0, -960.0),
        (120.0, -1608.0, 23, 13, 120.0, -1576.0),
    ]
    sector_lines = {45: [object()]}
    sectors = {45: (304, 368)}
    sector_tags = {45: 3}
    exit_points = [(0.0, 0.0, 11)]

    @staticmethod
    def _contains(point, lines):
        return point == (1504.0, -960.0)


def test_tagged_key_prefers_its_matching_progression_switch():
    navigator = CampaignNavigator(TaggedKeyMap())
    navigator.activated_switches.add(TaggedKeyMap.switch_points[1])
    player = SimpleNamespace(position_x=1504.0, position_y=-1008.0)
    key = SimpleNamespace(name="BlueCard", position_x=1504.0, position_y=-960.0)
    state = SimpleNamespace(objects=[key])

    assert navigator._choose_objective(state, player, 100.0, 100.0) == (
        1664.0,
        -992.0,
        "switch:23:3",
    )
    assert navigator._switch_required_by_visible_key(
        state, TaggedKeyMap.switch_points[0]
    )
    assert not navigator._switch_required_by_visible_key(
        state, TaggedKeyMap.switch_points[1]
    )


def test_gate_settle_catchup_preserves_later_sector_progress():
    navigator = CampaignNavigator(TaggedKeyMap())
    navigator.route.extend([
        Portal(310, 10.0, 0.0, 0),
        Portal(56, 20.0, 0.0, 0),
        Portal(57, 30.0, 0.0, 0),
        Portal(-1, 40.0, 0.0, 0),
    ])
    navigator.route_source_sector = 43

    assert navigator._catch_up_route(56, after_target=310)
    assert navigator.route_source_sector == 56
    assert [portal.target_sector for portal in navigator.route] == [57, -1]


def test_portal_stage_brakes_forward_and_lateral_momentum():
    player = SimpleNamespace(
        angle=0.0, velocity_x=5.0, velocity_y=3.0,
    )

    actions, forward_speed, lateral_speed = CampaignNavigator._motion_brake_actions(
        player
    )

    assert actions == {"move backward", "strafe left"}
    assert forward_speed == 5.0
    assert lateral_speed == 3.0


def test_portal_stage_tolerance_accepts_observed_engine_brake_quantum():
    assert 1.468 < PORTAL_STAGE_SPEED_TOLERANCE < 2.0
