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


def test_sector_center_points_into_open_portal_target() -> None:
    wad = Path(__file__).resolve().parents[3] / "dwasm-sim" / "wasm" / "fs" / "freedoom1.wad"
    parsed = WadMap(wad, "E1M1")
    portal = next(item for item in parsed.graph[57]
                  if item.target_sector == 101 and item.x == 1264.0)
    target_x, target_y = parsed.sector_centers[portal.target_sector]
    assert target_x > portal.x
    assert abs(target_y - portal.y) < 64.0
    probe = (portal.x + portal.normal_x * 32.0,
             portal.y + portal.normal_y * 32.0)
    # Sector 101 is thinner than one multi-tic movement. The directed probe
    # lands in its immediate planned successor, proving it crossed 101 rather
    # than the unrelated sector 144 seen with midpoint-heading navigation.
    probe_sector = parsed.sector_at(*probe)
    assert probe_sector != 57
    assert any(edge.target_sector == probe_sector for edge in parsed.graph[101])
    source_probe = (portal.x - portal.normal_x * 24.0,
                    portal.y - portal.normal_y * 24.0)
    assert parsed.sector_at(*source_probe) == 57


def test_blue_key_route_uses_lifts_not_impossible_ledge() -> None:
    wad = Path(__file__).resolve().parents[3] / "dwasm-sim" / "wasm" / "fs" / "freedoom1.wad"
    parsed = WadMap(wad, "E1M1")
    route = parsed.route((1888.0, 464.0), (2192.0, 576.0))
    assert route
    assert any(portal.special == 62 for portal in route)
    assert all(portal.floor_delta <= 32 or portal.special != 0 for portal in route)


def test_failed_portal_midpoint_selects_alternate_sector_entrance() -> None:
    wad = Path(__file__).resolve().parents[3] / "dwasm-sim" / "wasm" / "fs" / "freedoom1.wad"
    parsed = WadMap(wad, "E1M1")
    route = parsed.route((1200.0, -416.0), (1584.0, -508.0),
                         blocked_points=[(1264.0, -448.0)])
    assert route
    assert (route[0].x, route[0].y) != (1264.0, -448.0)
