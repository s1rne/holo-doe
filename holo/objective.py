"""Целевая функция, повторяющая проверку: 8-битная фаза, нормировка по максимуму, округление
восстановления до uint8, CC и SSIM по каждой плоскости. Округления пропускают градиент
через straight-through estimator (STE)."""

from dataclasses import dataclass

import torch

from .losses import pearson, ssim
from .optics import propagate, to_unit


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def quantize_phase(phase: torch.Tensor) -> torch.Tensor:
    """Фаза после записи в PNG и чтения обратно (phase_to_png → png_to_phase), с STE."""
    wrapped = torch.remainder(phase + torch.pi, 2 * torch.pi) - torch.pi
    code = ste_round((wrapped + torch.pi) / (2 * torch.pi) * 255)
    return code / 255 * 2 * torch.pi - torch.pi


def normalize(image: torch.Tensor, power: float = 0.0) -> torch.Tensor:
    """Нормировка по максимуму, как to_unit. При power > 0 значение то же, а градиент идёт через
    степенное среднее (mean x^p)^(1/p): его получают все яркие пиксели, а не один argmax."""
    if power <= 0:
        return to_unit(image)
    peak = image.amax(dim=(-2, -1), keepdim=True)
    soft = (image / peak.detach()).pow(power).mean(dim=(-2, -1), keepdim=True).pow(1 / power) * peak.detach()
    soft = soft * (peak / soft).detach()
    return image / soft.clamp_min(1e-12)


def reconstruct(phase: torch.Tensor, chirps: torch.Tensor, mode: str = "intensity", quantize: bool = True,
                power: float = 0.0) -> torch.Tensor:
    """Восстановление в 0..1 так, как оно попадает в PNG. mode: intensity |K|² или amplitude |K|."""
    if quantize:
        phase = quantize_phase(phase)
    field = propagate(phase, chirps).abs()
    unit = normalize(field**2 if mode == "intensity" else field, power)
    return ste_round(unit * 255) / 255 if quantize else unit


@dataclass(frozen=True)
class Objective:
    """ssim_weight: вес SSIM относительно CC; softmin: температура мягкого минимума по плоскостям
    (0 = среднее); modes: определения восстановления, по которым усредняется потеря;
    peak_power: степень мягкого максимума для градиента нормировки (0 = жёсткий максимум)."""

    ssim_weight: float = 1.0
    softmin: float = 0.0
    quantize: bool = True
    modes: tuple[str, ...] = ("intensity",)
    peak_power: float = 0.0

    def scores(self, phase: torch.Tensor, chirps: torch.Tensor, targets: torch.Tensor, mode: str = "intensity"):
        recon = reconstruct(phase, chirps, mode, self.quantize, self.peak_power)
        targets = targets.expand_as(recon)
        return pearson(recon, targets), ssim(recon, targets)

    def __call__(self, phase: torch.Tensor, chirps: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Потеря на каждый элемент батча, форма (...,)."""
        losses = []
        for mode in self.modes:
            cc, ss = self.scores(phase, chirps, targets, mode)
            score = (cc + self.ssim_weight * ss) / (1 + self.ssim_weight)
            if self.softmin > 0:
                n = torch.tensor(score.shape[-1], dtype=score.dtype).log()
                score = -self.softmin * (torch.logsumexp(-score / self.softmin, dim=-1) - n)
            else:
                score = score.mean(-1)
            losses.append(1 - score)
        return torch.stack(losses).mean(0)
