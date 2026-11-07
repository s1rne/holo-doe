"""Банки для обучения (data/cache, не в git) и отложенные проверочные наборы (data/heldout).

Обучение: кадры из DIV2K train HR и символы шрифтами из TRAIN_FONTS.
Проверка: Kodak (24 фото → 6 четвёрок), символы шрифтом Arial Bold, смешанные четвёрки.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from holo.dataset import N, glyph_bank, natural_crops, render_glyph

ROOT = Path(__file__).resolve().parent.parent / "data"


def save_set(folder: Path, images: list[np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images, 1):
        Image.fromarray(image, mode="L").save(folder / f"{i}.png")


def kodak_square(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    side = min(image.size)
    left, top = (image.size[0] - side) // 2, (image.size[1] - side) // 2
    pixels = np.asarray(image.crop((left, top, left + side, top + side)).resize((N, N), Image.LANCZOS), dtype=np.float64)
    return np.round((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--div2k", type=Path, default=ROOT / "raw" / "DIV2K_train_HR")
    parser.add_argument("--kodak", type=Path, default=ROOT / "raw" / "kodak")
    parser.add_argument("--per-image", type=int, default=40)
    parser.add_argument("--glyphs", type=int, default=30000)
    args = parser.parse_args()

    cache = ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    natural = natural_crops(sorted(args.div2k.glob("*.png")), args.per_image, np.random.default_rng(0))
    np.save(cache / "natural.npy", natural)
    np.save(cache / "glyphs.npy", glyph_bank(args.glyphs, seed=0))
    print(f"natural {natural.shape}, glyphs {args.glyphs}")

    heldout = ROOT / "heldout"
    kodak = [kodak_square(p) for p in sorted(args.kodak.glob("*.png"))]
    for k in range(len(kodak) // 4):
        save_set(heldout / f"kodak_{k + 1}", kodak[4 * k:4 * k + 4])
    rng = random.Random(1)
    glyphs = [render_glyph(rng, ["Arial Bold.ttf"]) for _ in range(16)]
    for k in range(4):
        save_set(heldout / f"glyphs_{k + 1}", glyphs[4 * k:4 * k + 4])
    for k in range(2):
        save_set(heldout / f"mixed_{k + 1}", [kodak[k], glyphs[k], kodak[k + 12], glyphs[k + 8]])


if __name__ == "__main__":
    main()
