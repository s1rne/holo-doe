"""Целевая функция, повторяющая проверку: 8-битная фаза, нормировка по максимуму, округление
восстановления до uint8, CC и SSIM по каждой плоскости. Округления пропускают градиент
через straight-through estimator (STE)."""

from dataclasses import dataclass

import torch

from .losses import pearson, speckle_excess, ssim
from .optics import to_unit
from .optics import reconstruct as field_image


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


def reconstruct(phase: torch.Tensor, chirps: torch.Tensor, mode: str = "amplitude", quantize: bool = True,
                power: float = 0.0) -> torch.Tensor:
    """Восстановление в 0..1 так, как оно попадает в PNG: нормировка по максимуму и округление
    до 8 бит. mode: amplitude |K| (как в проверке) или intensity |K|²."""
    if quantize:
        phase = quantize_phase(phase)
    unit = normalize(field_image(phase, chirps, mode), power)
    return ste_round(unit * 255) / 255 if quantize else unit


@dataclass(frozen=True)
class Objective:
    """ssim_weight: вес SSIM относительно CC; softmin: температура мягкого минимума по плоскостям
    (0 = среднее); modes: определения восстановления, по которым усредняется потеря;
    peak_power: степень мягкого максимума для градиента нормировки (0 = жёсткий максимум);
    dim_weight: вес штрафа за среднюю яркость восстановления после нормировки. Нормировка делит
    на максимум, поэтому штраф выгоняет энергию в яркий пик, а фон опускается ниже половины
    младшего разряда и округляется в ноль. Это главный рычаг ИСС при сравнении по амплитуде."""

    ssim_weight: float = 1.0
    softmin: float = 0.0
    quantize: bool = True
    modes: tuple[str, ...] = ("amplitude",)
    peak_power: float = 0.0
    dim_weight: float = 0.0
    worst_weight: float = 1.0
    """Доля мягкого минимума в сумме по плоскостям при softmin > 0; остальное — обычное среднее."""
    speckle_weight: float = 0.0
    """Вес штрафа за спекл: превышение локального СКО восстановления над локальным СКО цели.

    Ручка гладкости решения, независимая от вырожденного яркого пика. Гладкая фаза в плоскости
    восстановления поднимает SSIM честно, без затемнения фона; ценой части КК."""
    mean_weight: float = 0.0
    """Вес штрафа за недобор средней яркости: relu(1 − mean(recon)/mean(target)) по плоскости.

    Обратный по смыслу к dim_weight. Нормировка по максимуму делает выгодным вырожденное решение:
    часть энергии уходит в один яркий пик, фон после деления на максимум падает ниже половины
    младшего разряда и округляется ровно в ноль, окна ИСС на фоне совпадают с целью. Метрика это
    засчитывает, но изображение выглядит хуже. Штраф требует, чтобы отношение максимума к средней
    яркости восстановления не превышало того же отношения у цели, и запрещает этот обход."""

    def scores(self, phase: torch.Tensor, chirps: torch.Tensor, targets: torch.Tensor, mode: str = "amplitude"):
        recon = reconstruct(phase, chirps, mode, self.quantize, self.peak_power)
        targets = targets.expand_as(recon)
        return pearson(recon, targets), ssim(recon, targets), recon

    def __call__(self, phase: torch.Tensor, chirps: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Потеря на каждый элемент батча, форма (...,)."""
        losses = []
        for mode in self.modes:
            cc, ss, recon = self.scores(phase, chirps, targets, mode)
            score = (cc + self.ssim_weight * ss) / (1 + self.ssim_weight)
            if self.dim_weight:
                score = score - self.dim_weight * recon.mean((-2, -1))
            if self.speckle_weight:
                score = score - self.speckle_weight * speckle_excess(recon, targets.expand_as(recon))
            if self.mean_weight:
                ratio = recon.mean((-2, -1)) / targets.expand_as(recon).mean((-2, -1)).clamp_min(1e-6)
                score = score - self.mean_weight * (1 - ratio).clamp_min(0)
            if self.softmin > 0:
                n = torch.tensor(score.shape[-1], dtype=score.dtype).log()
                soft = -self.softmin * (torch.logsumexp(-score / self.softmin, dim=-1) - n)
                score = (1 - self.worst_weight) * score.mean(-1) + self.worst_weight * soft
            else:
                score = score.mean(-1)
            losses.append(1 - score)
        return torch.stack(losses).mean(0)
