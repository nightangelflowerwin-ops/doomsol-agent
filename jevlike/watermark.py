"""Small, consistent in-frame producer mark for generated review footage."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jevlike.provenance import PRODUCER_TAG


def stamp_frame(frame: np.ndarray, text: str = PRODUCER_TAG) -> np.ndarray:
    """Stamp a readable producer tag without covering the Doom HUD."""
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=14)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x, y = 10, 10
    draw.rounded_rectangle((x - 5, y - 4, x + width + 5, y + height + 4),
                           radius=4, fill=(0, 0, 0, 150), outline=(185, 65, 230, 210))
    draw.text((x, y), text, font=font, fill=(245, 230, 255, 235))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))
