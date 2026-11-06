from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_targets(folder: Path) -> tuple[list[str], np.ndarray]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")
    if len(files) != 4:
        raise ValueError(f"{folder}: нужно 4 PNG, найдено {len(files)}")
    images = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in files])
    if images.shape[1:] != (128, 128):
        raise ValueError(f"{folder}: изображения должны быть 128×128, получено {images.shape[1:]}")
    return [p.stem for p in files], images


def save_png(path: Path, image: np.ndarray | torch.Tensor) -> None:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(path)
