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


@dataclass(frozen=True)
class Portal:
    target_sector: int
    x: float
    y: float
    special: int


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
        self.sector_lines: dict[int, list[tuple[float, float, float, float]]] = collections.defaultdict(list)
        self.graph: dict[int, list[Portal]] = collections.defaultdict(list)
        self.exit_points: list[tuple[float, float, int]] = []
        self._parse_lines(lumps["LINEDEFS"])
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
            v1, v2, flags, special, _, right, left = struct.unpack_from("<HHHHHhh", data, offset)
            x1, y1 = self.vertices[v1]
            x2, y2 = self.vertices[v2]
            sectors = []
            for side in (right, left):
                if side >= 0:
                    sector = self.sides[side]
                    sectors.append(sector)
                    self.sector_lines[sector].append((x1, y1, x2, y2))
            if special in EXIT_SPECIALS:
                self.exit_points.append(((x1 + x2) / 2, (y1 + y2) / 2, special))
            if len(sectors) == 2 and sectors[0] != sectors[1] and not flags & 1:
                midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
                self.graph[sectors[0]].append(Portal(sectors[1], *midpoint, special))
                self.graph[sectors[1]].append(Portal(sectors[0], *midpoint, special))

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

    def route(self, start: tuple[float, float], goal: tuple[float, float],
              blocked_points: list[tuple[float, float]] | None = None) -> list[Portal]:
        source = self.sector_at(*start)
        target = self.sector_at(*goal)
        if source is None or target is None:
            return []
        if source == target:
            return [Portal(target, goal[0], goal[1], 0)]
        queue = collections.deque([source])
        previous: dict[int, tuple[int, Portal] | None] = {source: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for portal in self.graph[current]:
                if blocked_points and any(
                    math.hypot(portal.x - x, portal.y - y) < 64
                    for x, y in blocked_points
                ):
                    continue
                if portal.target_sector not in previous:
                    previous[portal.target_sector] = (current, portal)
                    queue.append(portal.target_sector)
        if target not in previous:
            if blocked_points:
                return self.route(start, goal)
            return []
        portals = []
        current = target
        while previous[current] is not None:
            parent, portal = previous[current]
            portals.append(portal)
            current = parent
        portals.reverse()
        portals.append(Portal(target, goal[0], goal[1], 0))
        return portals
