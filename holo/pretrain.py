"""Первый этап синтеза: подгонка ДОЭ под комплексное поле с гладкой фазой в плоскостях.

Метрики считаются по амплитуде |K|, поэтому спекл-шум в тёмных местах бьёт по ИСС прямо:
фон перестаёт округляться в ноль. Спекл возникает, когда фаза в плоскости восстановления
случайна. Здесь фаза плоскостей — свободный параметр низкого разрешения (scale×scale с
бикубической интерполяцией), и ДОЭ подгоняется под комплексную цель A·exp(jφ). Решение
выходит с гладкой фазой и малым спеклом; дальше его дотягивает градиентный спуск по метрикам.
"""

import torch
import torch.nn.functional as F

from .baselines import random_phase
from .optics import propagate
from .optimize import device


def smooth_target_fit(targets: torch.Tensor, chirps: torch.Tensor, scale: int = 4, steps: int = 3000,
                      lr: float = 0.3, phase_lr: float = 0.05, seed: int = 0) -> torch.Tensor:
    """Возвращает фазу ДОЭ (N, N). targets (P, N, N) — амплитуды целей в 0..1."""
    dev = device()
    chirps = chirps.to(torch.complex64).to(dev)
    amplitude = targets.float().to(dev)
    phase = torch.nn.Parameter(random_phase(targets.shape[-2:], seed).float().to(dev))
    low = torch.nn.Parameter(torch.zeros(targets.shape[0], 1, scale, scale, device=dev))
    opt = torch.optim.Adam([{"params": [phase], "lr": lr}, {"params": [low], "lr": phase_lr}])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        field = propagate(phase, chirps)
        image_phase = F.interpolate(low, size=targets.shape[-2:], mode="bicubic", align_corners=False)[:, 0]
        scaling = ((field.abs() * amplitude).sum((-2, -1), keepdim=True)
                   / (amplitude * amplitude).sum((-2, -1), keepdim=True).clamp_min(1e-9))
        loss = ((field - scaling * torch.polar(amplitude, image_phase)).abs() ** 2).mean()
        loss.backward()
        opt.step()
        schedule.step()
    return phase.detach().cpu().double()
