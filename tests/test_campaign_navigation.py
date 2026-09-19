from pathlib import Path
from types import SimpleNamespace
import sys


DOOM_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "doom"
sys.path.insert(0, str(DOOM_EXAMPLES))

from play_campaign_oracle import CampaignNavigator  # noqa: E402


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
