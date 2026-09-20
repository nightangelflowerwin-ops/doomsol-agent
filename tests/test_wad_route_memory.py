from collections import defaultdict
from pathlib import Path
import sys


DOOM_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "doom"
sys.path.insert(0, str(DOOM_EXAMPLES))

from wad_navigation import Portal, WadMap  # noqa: E402


def test_persistent_portal_penalty_selects_alternate_route():
    map_truth = object.__new__(WadMap)
    map_truth.graph = defaultdict(list, {
        0: [Portal(1, 10.0, 0.0, 0), Portal(2, 0.0, 10.0, 0)],
        1: [Portal(3, 20.0, 0.0, 0)],
        2: [Portal(3, 0.0, 20.0, 0)],
    })
    map_truth.sector_tags = defaultdict(int)
    map_truth.sector_at = lambda x, y: 0 if x == 0 else 3

    route = map_truth.route(
        (0.0, 0.0), (100.0, 100.0),
        edge_penalties={(0, 1, 10.0, 0.0): 6.0},
    )

    assert [portal.target_sector for portal in route[:-1]] == [2, 3]
