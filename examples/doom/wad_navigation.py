"""Minimal Doom-format WAD map parser and sector-portal route planner."""

from __future__ import annotations

import collections
import math
import struct
from dataclasses import dataclass
from pathlib import Path


EXIT_SPECIALS = {11, 51, 52, 124}
DOOR_SPECIALS = {
    1, 2, 3, 4, 16, 26, 27, 28, 31, 32, 33, 34, 42, 46, 61, 63,
    75, 76, 86, 90, 99, 103, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118,
}
KEY_THING_TYPES = {5: "BlueCard", 6: "YellowCard", 13: "RedCard",
                   38: "RedSkull", 39: "YellowSkull", 40: "BlueSkull"}
# Classic Doom locked-door specials.  These are the authoritative equivalent
# of the colored key signs painted beside/in the door texture.
LOCKED_DOOR_KEYS = {
    26: "BlueCard", 32: "BlueCard", 99: "BlueCard", 133: "BlueCard",
    27: "YellowCard", 34: "YellowCard", 136: "YellowCard",
    28: "RedCard", 33: "RedCard", 135: "RedCard",
}


@dataclass(frozen=True)
class Portal:
    target_sector: int
    x: float
    y: float
    special: int
    floor_delta: int = 0
    opening: int = 128
    normal_x: float = 0.0
    normal_y: float = 0.0
    required_key: str | None = None


class WadMap:
    def __init__(self, wad: Path, map_name: str) -> None:
        self.wad = Path(wad)
        self.map_name = map_name.upper()
        lumps = self._map_lumps()
        self.vertices = [
            struct.unpack_from("<hh", lumps["VERTEXES"], offset)
            for offset in range(0, len(lumps["VERTEXES"]), 4)
        ]
        self.sides = [
            struct.unpack_from("<hh8s8s8sH", lumps["SIDEDEFS"], offset)[-1]
            for offset in range(0, len(lumps["SIDEDEFS"]), 30)
        ]
        self.sectors = [
            struct.unpack_from("<hh8s8shhh", lumps["SECTORS"], offset)[:2]
            for offset in range(0, len(lumps["SECTORS"]), 26)
        ]
        self.sector_tags = [
            struct.unpack_from("<hh8s8shhh", lumps["SECTORS"], offset)[-1]
            for offset in range(0, len(lumps["SECTORS"]), 26)
        ]
        self.sector_lines: dict[int, list[tuple[float, float, float, float]]] = collections.defaultdict(list)
        self.solid_lines: list[tuple[float, float, float, float]] = []
        self.graph: dict[int, list[Portal]] = collections.defaultdict(list)
        self.transition_lines: dict[tuple[int, int], list[tuple[float, float, float, float]]] = collections.defaultdict(list)
        self.exit_points: list[tuple[float, float, int]] = []
        # activation_x/y, special, tag, linedef_midpoint_x/y
        self.switch_points: list[tuple[float, float, int, int, float, float]] = []
        self._parse_lines(lumps["LINEDEFS"])
        self.sector_centers = {
            sector: (
                sum(point[0] for line in lines for point in ((line[0], line[1]), (line[2], line[3]))) /
                (2 * len(lines)),
                sum(point[1] for line in lines for point in ((line[0], line[1]), (line[2], line[3]))) /
                (2 * len(lines)),
            )
            for sector, lines in self.sector_lines.items() if lines
        }
        self.key_points = self._parse_keys(lumps["THINGS"])

    def _map_lumps(self) -> dict[str, bytes]:
        data = self.wad.read_bytes()
        _, count, directory = struct.unpack_from("<4sii", data, 0)
        entries = []
        for index in range(count):
            offset, size, raw_name = struct.unpack_from("<ii8s", data, directory + index * 16)
            entries.append((raw_name.decode(errors="ignore").rstrip(chr(0)), offset, size))
        marker = next(index for index, entry in enumerate(entries) if entry[0].upper() == self.map_name)
        result = {}
        for name, offset, size in entries[marker + 1:marker + 12]:
            result[name] = data[offset:offset + size]
        return result

    def _parse_lines(self, data: bytes) -> None:
        for offset in range(0, len(data), 14):
            v1, v2, flags, special, tag, right, left = struct.unpack_from("<HHHHHhh", data, offset)
            x1, y1 = self.vertices[v1]
            x2, y2 = self.vertices[v2]
            sectors = []
            for side in (right, left):
                if side >= 0:
                    sector = self.sides[side]
                    sectors.append(sector)
                    self.sector_lines[sector].append((x1, y1, x2, y2))
            if len(sectors) < 2:
                self.solid_lines.append((x1, y1, x2, y2))
            if special in EXIT_SPECIALS:
                self.exit_points.append(((x1 + x2) / 2, (y1 + y2) / 2, special))
            # Switch-once floor actions are progression controls rather than
            # traversable portals. Preserve their authoritative map position
            # and tag so the controller can operate them before routing across
            # the sectors they unlock (E1M1 uses special 23 / tag 3).
            if special in {23}:
                midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2
                length = math.hypot(x2 - x1, y2 - y1) or 1.0
                # A one-sided switch is usable from its right/front sidedef.
                activation_x = midpoint_x + (y2 - y1) / length * 32.0
                activation_y = midpoint_y - (x2 - x1) / length * 32.0
                self.switch_points.append((activation_x, activation_y,
                                           special, tag,
                                           midpoint_x, midpoint_y))
            if len(sectors) == 2 and sectors[0] != sectors[1] and not flags & 1:
                midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
                length = math.hypot(x2 - x1, y2 - y1) or 1.0
                # Doom SIDEDEFS are ordered right/front then left/back. These
                # are the directed unit normals for each graph direction.
                right_to_left = (-(y2 - y1) / length, (x2 - x1) / length)
                left_to_right = ((y2 - y1) / length, -(x2 - x1) / length)
                floor_a, ceiling_a = self.sectors[sectors[0]]
                floor_b, ceiling_b = self.sectors[sectors[1]]
                opening = min(ceiling_a, ceiling_b) - max(floor_a, floor_b)
                self.graph[sectors[0]].append(Portal(
                    sectors[1], *midpoint, special, floor_b - floor_a, opening,
                    *right_to_left, LOCKED_DOOR_KEYS.get(special),
                ))
                self.transition_lines[(sectors[0], sectors[1])].append(
                    (x1, y1, x2, y2)
                )
                self.graph[sectors[1]].append(Portal(
                    sectors[0], *midpoint, special, floor_a - floor_b, opening,
                    *left_to_right, LOCKED_DOOR_KEYS.get(special),
                ))
                self.transition_lines[(sectors[1], sectors[0])].append(
                    (x1, y1, x2, y2)
                )

    def _parse_keys(self, data: bytes) -> list[tuple[float, float, str]]:
        result = []
        for offset in range(0, len(data), 10):
            x, y, _, thing_type, _ = struct.unpack_from("<hhhhh", data, offset)
            if thing_type in KEY_THING_TYPES:
                result.append((float(x), float(y), KEY_THING_TYPES[thing_type]))
        return result

    @staticmethod
    def _contains(point: tuple[float, float], lines) -> bool:
        x, y = point
        crossings = 0
        for x1, y1, x2, y2 in lines:
            if (y1 > y) == (y2 > y):
                continue
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                crossings += 1
        return bool(crossings % 2)

    def sector_at(self, x: float, y: float) -> int | None:
        for sector, lines in self.sector_lines.items():
            if self._contains((x, y), lines):
                return sector
        return None

    def local_path(self, start: tuple[float, float], goal: tuple[float, float],
                   sector: int, grid: int = 8,
                   avoid_targets: set[int] | None = None) -> list[tuple[float, float]]:
        """Find a small collision-aware path through one concave sector.

        The sector graph cannot describe interior walls: two points can share
        a sector while the straight segment between them crosses solid map
        geometry.  A bounded grid search supplies only the missing intra-room
        waypoints and leaves inter-sector routing unchanged.
        """
        lines = self.sector_lines.get(sector, [])
        if not lines:
            return []
        xs = [value for line in lines for value in (line[0], line[2])]
        ys = [value for line in lines for value in (line[1], line[3])]
        min_x = math.floor(min(xs) / grid) * grid
        max_x = math.ceil(max(xs) / grid) * grid
        min_y = math.floor(min(ys) / grid) * grid
        max_y = math.ceil(max(ys) / grid) * grid
        def distance_to_line(point, line) -> float:
            x, y = point
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            denominator = dx * dx + dy * dy
            t = 0.0 if not denominator else max(0.0, min(
                1.0, ((x - x1) * dx + (y - y1) * dy) / denominator
            ))
            return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))

        avoided_lines = [
            line for target in (avoid_targets or set())
            for line in self.transition_lines.get((sector, target), [])
        ]
        nodes = {
            (float(x), float(y))
            for x in range(min_x, max_x + grid, grid)
            for y in range(min_y, max_y + grid, grid)
            if self.sector_at(float(x), float(y)) == sector and
            all(distance_to_line((float(x), float(y)), line) >= 24.0
                for line in self.solid_lines) and
            all(distance_to_line((float(x), float(y)), line) >= 8.0
                for line in avoided_lines)
        }
        if not nodes:
            return []
        source = min(nodes, key=lambda point: math.hypot(
            point[0] - start[0], point[1] - start[1]
        ))
        target = min(nodes, key=lambda point: math.hypot(
            point[0] - goal[0], point[1] - goal[1]
        ))
        queue = collections.deque([source])
        previous: dict[tuple[float, float], tuple[float, float] | None] = {
            source: None
        }
        offsets = (
            (grid, 0), (-grid, 0), (0, grid), (0, -grid),
            (grid, grid), (grid, -grid), (-grid, grid), (-grid, -grid),
        )
        while queue and target not in previous:
            point = queue.popleft()
            for dx, dy in offsets:
                neighbor = (point[0] + dx, point[1] + dy)
                if neighbor in nodes and neighbor not in previous:
                    previous[neighbor] = point
                    queue.append(neighbor)
        if target not in previous:
            return []
        path = []
        point: tuple[float, float] | None = target
        while point is not None:
            path.append(point)
            point = previous[point]
        path.reverse()
        # Keep direction changes, not every 16-unit grid cell.
        compressed = [path[0]]
        last_direction = None
        for index in range(1, len(path)):
            direction = (
                path[index][0] - path[index - 1][0],
                path[index][1] - path[index - 1][1],
            )
            if last_direction is not None and direction != last_direction:
                compressed.append(path[index - 1])
            last_direction = direction
        compressed.append(path[-1])
        return compressed

    def route(self, start: tuple[float, float], goal: tuple[float, float],
              blocked_points: list[tuple[float, float]] | None = None,
              blocked_edges: set[tuple[int, int, float, float]] | None = None,
              unlocked_tags: set[int] | None = None) -> list[Portal]:
        source = self.sector_at(*start)
        target = self.sector_at(*goal)
        if source is None or target is None:
            return []
        if source == target:
            # The goal coordinate is not a sector boundary. Marking it with
            # the containing sector makes the controller run gate-confirmation
            # logic at pickups and overshoot them.
            return [Portal(-1, goal[0], goal[1], 0)]
        queue = collections.deque([source])
        previous: dict[int, tuple[int, Portal] | None] = {source: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for portal in self.graph[current]:
                # A two-sided linedef is not necessarily a walkable step.
                # Reject tall, untagged ledges; tagged specials remain eligible
                # because they can represent lifts, stairs, or other movers.
                dynamically_unlocked = bool(
                    unlocked_tags and (
                        self.sector_tags[current] in unlocked_tags or
                        self.sector_tags[portal.target_sector] in unlocked_tags
                    )
                )
                if (portal.floor_delta > 32 and portal.special == 0 and
                        not dynamically_unlocked):
                    continue
                if blocked_points and any(
                    math.hypot(portal.x - x, portal.y - y) < 64
                    for x, y in blocked_points
                ):
                    continue
                if blocked_edges and (
                    current, portal.target_sector, float(portal.x), float(portal.y)
                ) in blocked_edges:
                    continue
                if portal.target_sector not in previous:
                    previous[portal.target_sector] = (current, portal)
                    queue.append(portal.target_sector)
        if target not in previous:
            return []
        portals = []
        current = target
        while previous[current] is not None:
            parent, portal = previous[current]
            portals.append(portal)
            current = parent
        portals.reverse()
        portals.append(Portal(-1, goal[0], goal[1], 0))
        return portals
