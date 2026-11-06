"""Тестовые наборы по 4 изображения 128×128: полутоновые фотографии, бинарные символы и
дополнительные снимки другого происхождения (кот, ракета, Луна, микроскопия клетки)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data, transform

ROOT = Path(__file__).resolve().parent.parent / "data" / "targets"


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image[..., :3] @ np.array([0.299, 0.587, 0.114])
    image = transform.resize(image, (128, 128), anti_aliasing=True)
    image = (image - image.min()) / (image.max() - image.min())
    return np.round(image * 255).astype(np.uint8)


def glyph(char: str) -> np.ndarray:
    canvas = Image.new("L", (128, 128), 0)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 96)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), char, font=font)
    draw.text(((128 - box[2] - box[0]) / 2, (128 - box[3] - box[1]) / 2), char, fill=255, font=font)
    return np.asarray(canvas)


def main() -> None:
    sets = {
        "photos": {"1_camera": data.camera(), "2_astronaut": data.astronaut(), "3_coffee": data.coffee(), "4_coins": data.coins()},
        "glyphs": {f"{i}_{c}": None for i, c in enumerate("ЛАЗР", 1)},
        "extra": {"1_chelsea": data.chelsea(), "2_rocket": data.rocket(), "3_moon": data.moon(), "4_cell": data.cell()},
    }
    for name, images in sets.items():
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        for key, image in images.items():
            pixels = glyph(key.split("_")[1]) if image is None else gray(image)
            Image.fromarray(pixels, mode="L").save(folder / f"{key}.png")


if __name__ == "__main__":
    main()
