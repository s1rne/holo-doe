"""Оценка ДОЭ строго по условию: фаза кодируется в 8-битный PNG, восстановление считается из этого PNG."""

import json
from pathlib import Path

import numpy as np
import torch

from .io import save_png
from .metrics import report
from .optics import Setup, phase_to_png, png_to_phase, reconstruct, to_uint8


def evaluate(phase: torch.Tensor, targets: np.ndarray, setup: Setup = Setup(),
             mode: str = "amplitude") -> tuple[np.ndarray, np.ndarray, dict]:
    code = phase_to_png(phase.cpu().double())
    recon = to_uint8(reconstruct(png_to_phase(code), setup.chirps(), mode)).numpy()
    return code.numpy(), recon, {**report(targets, recon), "mode": mode}


def save(out: Path, names: list[str], code: np.ndarray, recon: np.ndarray, metrics: dict, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_png(out / "doe.png", code)
    for name, image, z in zip(names, recon, Setup().distances):
        save_png(out / f"recon_{name}_{z * 1e3:.1f}mm.png", image)
    (out / "metrics.json").write_text(json.dumps({**metrics, **extra, "files": names}, indent=2, ensure_ascii=False))
