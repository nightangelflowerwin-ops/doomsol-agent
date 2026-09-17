from pathlib import Path

from wad_navigation import WadMap


def test_freedoom_e1m1_exposes_campaign_objectives() -> None:
    wad = Path(__file__).resolve().parents[3] / "dwasm-sim" / "wasm" / "fs" / "freedoom1.wad"
    parsed = WadMap(wad, "E1M1")
    assert len(parsed.sector_lines) == 182
    assert parsed.exit_points == [(-400.0, 1296.0, 11)]
    assert any(name == "BlueCard" for _, _, name in parsed.key_points)


def test_freedoom_e1m1_has_route_from_spawn_to_exit() -> None:
    wad = Path(__file__).resolve().parents[3] / "dwasm-sim" / "wasm" / "fs" / "freedoom1.wad"
    parsed = WadMap(wad, "E1M1")
    route = parsed.route((-416.0, 256.0), parsed.exit_points[0][:2])
    assert route
    assert (route[-1].x, route[-1].y) == parsed.exit_points[0][:2]
