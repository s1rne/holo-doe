"""Отложенные наборы для проверки обобщения: четвёрка похожих изображений и четвёрка разных.

Перекрёстные помехи между плоскостями зависят от того, насколько изображения похожи: близкие
картинки хуже разводятся по плоскостям, непохожие — лучше. Поэтому к отложенным наборам
(`data/heldout`, см. docs/feedforward.md) добавляются два крайних случая.

- `similar_glyphs` — О, Ф, Q, D шрифтом Arial Bold: замкнутые округлые формы одного размера.
- `similar_photos` — четыре перекрывающихся кадра одного снимка.
- `diverse` — фотография, бинарный символ, редкие точки и мелкая периодическая текстура.

Ни одно из изображений не встречается в обучении: банк фото собран из DIV2K, банк символов —
шрифтами из `holo.dataset.TRAIN_FONTS`, куда Arial Bold не входит.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from skimage import data, transform

ROOT = Path(__file__).resolve().parent.parent / "data" / "heldout"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def stretch(image: np.ndarray) -> np.ndarray:
    """В оттенки серого, к размеру 128×128 и на весь диапазон 0..255."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = image[..., :3] @ np.array([0.299, 0.587, 0.114])
    if image.shape != (128, 128):
        image = transform.resize(image, (128, 128), anti_aliasing=True)
    image = (image - image.min()) / max(image.max() - image.min(), 1e-12)
    return np.round(image * 255).astype(np.uint8)


def glyph(char: str, size: int = 96) -> np.ndarray:
    canvas = Image.new("L", (128, 128), 0)
    font = ImageFont.truetype(FONT, size)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), char, font=font)
    draw.text(((128 - box[2] - box[0]) / 2, (128 - box[3] - box[1]) / 2), char, fill=255, font=font)
    return np.asarray(canvas)


def save_set(name: str, images: list[np.ndarray]) -> None:
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images, 1):
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(folder / f"{i}.png")
    print(f"{name}: {len(images)} изображений")


def main() -> None:
    save_set("similar_glyphs", [glyph(c) for c in "ОФQD"])

    photo = stretch(data.astronaut())
    offsets = [(0, 0), (24, 0), (0, 24), (24, 24)]
    save_set("similar_photos", [stretch(photo[y:y + 360, x:x + 360]) for x, y in offsets])

    rng = np.random.default_rng(0)
    dots = np.zeros((128, 128), dtype=np.uint8)
    index = rng.choice(128 * 128, 120, replace=False)
    dots.ravel()[index] = 255
    dots = np.asarray(Image.fromarray(dots).filter(ImageFilter.MaxFilter(3)))
    stripes = np.round(127.5 * (1 + np.sin(np.arange(128)[None, :] * 0.9 + np.arange(128)[:, None] * 0.5))).astype(np.uint8)
    save_set("diverse", [stretch(data.camera()), glyph("Ж"), dots, stripes])


if __name__ == "__main__":
    main()
