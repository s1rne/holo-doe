"""Обучающие сцены для обучаемого оптимизатора. Изображения тестовых наборов (photos, glyphs, extra)
в обучение не входят: только остальные картинки skimage и символы других алфавитов."""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import data

TRAIN_IMAGES = ("brick", "grass", "gravel", "horse", "immunohistochemistry", "retina", "page", "text", "clock",
                "shepp_logan_phantom", "hubble_deep_field", "binary_blobs", "colorwheel", "checkerboard")
TRAIN_CHARS = "ABDEFGHKMNQRSTUVWXYZ0123456789&%#@БГДЖИФШЮЯ"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = image[..., :3] @ np.array([0.299, 0.587, 0.114])
    return (image - image.min()) / max(image.max() - image.min(), 1e-12)


def glyph(char: str, size: int, rng: np.random.Generator) -> np.ndarray:
    canvas = Image.new("L", (128, 128), 0)
    font = ImageFont.truetype(FONT, size)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), char, font=font)
    shift = rng.integers(-12, 13, 2)
    draw.text(((128 - box[2] - box[0]) / 2 + shift[0], (128 - box[3] - box[1]) / 2 + shift[1]), char, fill=255, font=font)
    return np.asarray(canvas, dtype=np.float64) / 255


class TrainScenes:
    """Случайные четвёрки 128×128: кропы с поворотом и масштабом либо символы, в равных долях."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.pool = [gray(getattr(data, name)()) for name in TRAIN_IMAGES]

    def crop(self) -> np.ndarray:
        image = self.pool[self.rng.integers(len(self.pool))]
        side = int(self.rng.integers(128, min(image.shape) + 1))
        y, x = (self.rng.integers(0, s - side + 1) for s in image.shape)
        patch = Image.fromarray(image[y:y + side, x:x + side]).resize((128, 128), Image.BILINEAR)
        patch = np.rot90(np.asarray(patch), self.rng.integers(4))
        patch = (patch - patch.min()) / max(patch.max() - patch.min(), 1e-12)
        return np.round(patch * 255) / 255

    def scene(self) -> np.ndarray:
        if self.rng.random() < 0.5:
            return np.stack([self.crop() for _ in range(4)])
        chars = self.rng.choice(list(TRAIN_CHARS), 4, replace=False)
        return np.stack([glyph(c, int(self.rng.integers(70, 110)), self.rng) for c in chars])

    def batch(self, size: int) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.scene() for _ in range(size)])).float()
