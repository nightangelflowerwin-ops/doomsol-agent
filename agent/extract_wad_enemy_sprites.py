"""Extract living/attacking enemy sprite rotations from a Doom-format WAD."""
import argparse
import json
import struct
from pathlib import Path

from PIL import Image


# Frame letters intentionally stop before each monster's death animation so
# corpses are not taught as active threats.
ENEMY_FRAMES = {
    "POSS": set("ABCDEFG"),   # former human
    "SPOS": set("ABCDEFG"),   # shotgun human
    "TROO": set("ABCDEFGH"),  # imp
    "SARG": set("ABCDEFG"),   # demon/spectre
    "HEAD": set("ABC"),       # cacodemon
    "BOSS": set("ABCDEFG"),   # baron
    "BOS2": set("ABCDEFG"),   # hell knight where present
    "SKUL": set("ABCD"),      # lost soul
    "CYBR": set("ABCDEFG"),   # cyberdemon
    "SPID": set("ABCDEFG"),   # spider mastermind
}


def directory(blob):
    count, offset = struct.unpack_from("<II", blob, 4)
    rows = []
    for index in range(count):
        position, size, raw_name = struct.unpack_from("<II8s", blob, offset + index * 16)
        rows.append((raw_name.rstrip(b"\0").decode("ascii", "ignore").upper(), position, size))
    return rows


def decode_patch(payload, palette):
    if len(payload) < 8:
        return None
    width, height, left, top = struct.unpack_from("<HHhh", payload, 0)
    if not (0 < width <= 1024 and 0 < height <= 1024) or len(payload) < 8 + width * 4:
        return None
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = image.load()
    offsets = struct.unpack_from(f"<{width}I", payload, 8)
    for x, cursor in enumerate(offsets):
        previous_top = -1
        while cursor < len(payload):
            top_delta = payload[cursor]
            cursor += 1
            if top_delta == 255:
                break
            if cursor + 2 > len(payload):
                break
            length = payload[cursor]
            cursor += 2  # length plus unused byte
            if top_delta <= previous_top:
                top_delta += previous_top
            previous_top = top_delta
            for dy in range(length):
                if cursor + dy >= len(payload):
                    break
                y = top_delta + dy
                if 0 <= y < height:
                    colour = palette[payload[cursor + dy]]
                    pixels[x, y] = (*colour, 255)
            cursor += length + 1  # pixels plus trailing unused byte
    if image.getbbox() is None:
        return None
    return image, {"width": width, "height": height, "left_offset": left, "top_offset": top}


def extract(wad, output):
    blob = wad.read_bytes()
    entries = directory(blob)
    playpal = next((blob[pos:pos + size] for name, pos, size in entries if name == "PLAYPAL"), None)
    if not playpal or len(playpal) < 768:
        raise ValueError("WAD has no usable PLAYPAL palette")
    palette = [tuple(playpal[i:i + 3]) for i in range(0, 768, 3)]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, position, size in entries:
        family = name[:4]
        frame = name[4:5]
        if family not in ENEMY_FRAMES or frame not in ENEMY_FRAMES[family]:
            continue
        decoded = decode_patch(blob[position:position + size], palette)
        if not decoded:
            continue
        image, geometry = decoded
        destination = output / family.lower() / f"{name.lower()}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        records.append({"name": name, "family": family, "frame": frame,
                        "file": str(destination.relative_to(output)), **geometry})
    report = {"schema_version": 1, "source_wad": str(wad), "enemy_families": sorted(ENEMY_FRAMES),
              "sprite_count": len(records), "sprites": records}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wad", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.wad, args.output)
    print(json.dumps({k: report[k] for k in ("source_wad", "enemy_families", "sprite_count")}, indent=2))


if __name__ == "__main__":
    main()
