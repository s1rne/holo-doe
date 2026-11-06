"""Данные для обучения без голограмм-эталонов: банки изображений 128×128 и случайные четвёрки из них.

Два банка: кадры из натуральных фотографий (DIV2K) и синтетические символы, слова и фигуры.
Банки строятся один раз (scripts/make_dataset.py) и хранятся в uint8, аугментации делаются на лету.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

N = 128
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
# Arial Bold исключён: им нарисованы проверочные символы в data/targets/glyphs и data/heldout.
TRAIN_FONTS = [
    "Arial.ttf", "Arial Black.ttf", "Arial Italic.ttf", "Arial Narrow.ttf", "Arial Narrow Bold.ttf",
    "Comic Sans MS.ttf", "Comic Sans MS Bold.ttf", "Courier New.ttf", "Courier New Bold.ttf",
    "Georgia.ttf", "Georgia Bold.ttf", "Georgia Italic.ttf", "Impact.ttf", "Tahoma.ttf", "Tahoma Bold.ttf",
    "Times New Roman.ttf", "Times New Roman Bold.ttf", "Trebuchet MS.ttf", "Trebuchet MS Bold.ttf",
    "Verdana.ttf", "Verdana Bold.ttf", "Verdana Bold Italic.ttf",
]
CHARS = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def natural_crops(files: list[Path], per_image: int, rng: np.random.Generator) -> np.ndarray:
    """Случайные квадратные кадры разного масштаба, уменьшенные до 128×128; почти однотонные отбрасываются."""
    crops = []
    for path in files:
        image = Image.open(path).convert("L")
        short = min(image.size)
        found = 0
        for _ in range(per_image * 4):
            if found == per_image:
                break
            side = int(N * 2 ** rng.uniform(0, np.log2(short / N)))
            x, y = rng.integers(0, image.size[0] - side + 1), rng.integers(0, image.size[1] - side + 1)
            crop = image.crop((x, y, x + side, y + side)).resize((N, N), Image.LANCZOS)
            pixels = np.asarray(crop, dtype=np.uint8)
            if pixels.std() > 12:
                crops.append(pixels)
                found += 1
    return np.stack(crops)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


def render_glyph(rng: random.Random, fonts: list[str]) -> np.ndarray:
    """Символ, короткая надпись или набор фигур; яркость, фон, поворот и размытие случайны."""
    canvas = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(canvas)
    kind = rng.random()
    if kind < 0.7:
        length = 1 if kind < 0.45 else rng.randint(2, 4)
        text = "".join(rng.choice(CHARS) for _ in range(length))
        if length > 2 and rng.random() < 0.5:
            text = text[:2] + "\n" + text[2:]
        size = rng.randint(40, 110) if length == 1 else rng.randint(24, 60)
        font = _font(rng.choice(fonts), size)
        box = draw.multiline_textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        x = (N - w) / 2 - box[0] + rng.uniform(-0.3, 0.3) * max(N - w, 0)
        y = (N - h) / 2 - box[1] + rng.uniform(-0.3, 0.3) * max(N - h, 0)
        draw.multiline_text((x, y), text, fill=rng.randint(150, 255), font=font, align="center")
    else:
        for _ in range(rng.randint(1, 5)):
            x0, y0 = rng.uniform(0, N - 20), rng.uniform(0, N - 20)
            x1, y1 = x0 + rng.uniform(15, N - x0), y0 + rng.uniform(15, N - y0)
            fill = rng.randint(80, 255)
            shape = rng.random()
            if shape < 0.35:
                draw.ellipse((x0, y0, x1, y1), fill=fill)
            elif shape < 0.7:
                draw.rectangle((x0, y0, x1, y1), fill=fill)
            elif shape < 0.85:
                draw.polygon([(rng.uniform(0, N), rng.uniform(0, N)) for _ in range(3)], fill=fill)
            else:
                draw.line((x0, y0, x1, y1), fill=fill, width=rng.randint(3, 12))
    canvas = canvas.rotate(rng.uniform(-30, 30), resample=Image.BILINEAR)
    if rng.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2)))
    pixels = np.asarray(canvas, dtype=np.float32)
    if rng.random() < 0.15:
        pixels = 255 - pixels
    return np.round(pixels).astype(np.uint8)


def glyph_bank(count: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.stack([render_glyph(rng, TRAIN_FONTS) for _ in range(count)])


def procedural(count: int, device, n: int = N) -> torch.Tensor:
    """Процедурные сцены в 0..1: полосы, радиальные узоры и шум с ограниченной полосой.

    Банки фото и символов покрывают изображения с «естественным» спектром; процедурные сцены
    добавляют широкий спектр и резкие периодические структуры, на которых перекрёстные помехи
    между плоскостями ведут себя иначе."""
    axis = torch.linspace(-1, 1, n, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    out = torch.empty(count, n, n, device=device)
    kind = torch.randint(3, (count,), device=device)
    for i in range(count):
        if kind[i] == 0:
            freq = torch.empty(2, device=device).uniform_(-14, 14)
            phase = torch.rand(1, device=device) * 6.28
            image = torch.sin(freq[0] * x + freq[1] * y + phase)
            if torch.rand(1).item() < 0.5:
                image = image.sign()
        elif kind[i] == 1:
            cx, cy = torch.empty(2, device=device).uniform_(-0.6, 0.6)
            r = ((x - cx) ** 2 + (y - cy) ** 2).sqrt()
            image = torch.cos(torch.empty(1, device=device).uniform_(3, 24) * r
                              + torch.empty(1, device=device).uniform_(0, 8) * torch.atan2(y - cy, x - cx))
        else:
            cut = int(torch.empty(1).uniform_(3, 32).item())
            spectrum = torch.zeros(n, n, dtype=torch.complex64, device=device)
            box = torch.randn(2 * cut, 2 * cut, dtype=torch.complex64, device=device)
            spectrum[:2 * cut, :2 * cut] = box
            image = torch.fft.ifft2(spectrum).real
        low, high = image.amin(), image.amax()
        out[i] = (image - low) / (high - low).clamp_min(1e-6)
    return out


class QuadSampler:
    """Пакеты по B четвёрок из трёх банков (фото, символы, процедурные сцены).

    Четверть пакета — смешанные четвёрки, где каждое изображение берётся из своего банка:
    у жюри четыре картинки не обязаны быть однотипными. Аугментации: отражение,
    транспонирование, гамма, сжатие диапазона яркости, округление до 8 бит."""

    SHARES = (0.40, 0.20, 0.10, 0.30)
    """Доли: только фото, только символы, только процедурные, смесь."""

    def __init__(self, natural: np.ndarray, glyphs: np.ndarray, device, shares: tuple[float, ...] | None = None):
        self.banks = [torch.from_numpy(natural).to(device), torch.from_numpy(glyphs).to(device)]
        self.shares = torch.tensor(shares or self.SHARES, device=device)
        self.device = device

    def sample(self, batch: int) -> torch.Tensor:
        theme = torch.multinomial(self.shares, batch, replacement=True)
        source = theme[:, None].expand(batch, 4).clone()
        mixed = theme == 3
        source[mixed] = torch.randint(3, (int(mixed.sum()), 4), device=self.device)
        out = torch.empty(batch, 4, N, N, device=self.device)
        for bank in (0, 1):
            mask = source == bank
            count = int(mask.sum())
            if count:
                index = torch.randint(len(self.banks[bank]), (count,), device=self.device)
                out[mask] = self.banks[bank][index].float() / 255
        mask = source == 2
        count = int(mask.sum())
        if count:
            out[mask] = procedural(count, self.device)
        return self._augment(out)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = torch.where(torch.rand(b, 4, 1, 1, device=self.device) < 0.5, x.flip(-1), x)
        x = torch.where(torch.rand(b, 4, 1, 1, device=self.device) < 0.5, x.transpose(-1, -2), x)
        gamma = torch.exp(torch.empty(b, 4, 1, 1, device=self.device).uniform_(-0.5, 0.5))
        x = x.clamp_min(0) ** gamma
        low = x.amin(dim=(-2, -1), keepdim=True)
        high = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - low) / (high - low).clamp_min(1e-6)
        # Часть целей не растянута на весь диапазон 0..255.
        squeeze = torch.rand(b, 4, 1, 1, device=self.device) < 0.3
        low = torch.empty(b, 4, 1, 1, device=self.device).uniform_(0, 0.3)
        high = torch.empty(b, 4, 1, 1, device=self.device).uniform_(0.7, 1)
        x = torch.where(squeeze, low + (high - low) * x, x)
        # Как при сохранении цели в 8 бит.
        return torch.round(x * 255) / 255
