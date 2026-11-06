"""Итеративные эталоны, с которыми сравнивается нейросетевой метод."""

import torch

from .losses import image_loss
from .optics import propagate, reconstruct, to_unit


def back_propagate(field: torch.Tensor, chirps: torch.Tensor) -> torch.Tensor:
    """Обратный переход из плоскостей восстановления в плоскость ДОЭ, обратный к propagate."""
    doe = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    return doe * chirps.conj()


def random_phase(shape, seed: int, device=None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return ((torch.rand(shape, generator=generator, dtype=torch.float64) * 2 - 1) * torch.pi).to(device)


def gerchberg_saxton(targets: torch.Tensor, chirps: torch.Tensor, iterations: int = 200, seed: int = 0) -> torch.Tensor:
    """Многоплоскостной GS: в каждой плоскости амплитуда заменяется целевой, поля плоскостей суммируются в ДОЭ."""
    amplitude = targets
    phase = random_phase(targets.shape[-2:], seed, targets.device)
    for _ in range(iterations):
        field = propagate(phase, chirps)
        scale = field.abs().mean(dim=(-2, -1), keepdim=True) / amplitude.mean(dim=(-2, -1), keepdim=True)
        constrained = torch.polar(amplitude * scale, field.angle())
        phase = back_propagate(constrained, chirps).sum(0).angle()
    return phase


def ping_pong(targets: torch.Tensor, chirps: torch.Tensor, iterations: int = 200, seed: int = 0) -> torch.Tensor:
    """Последовательный ping-pong по Дорша и Маковского: за один проход плоскости обходятся по
    очереди, в каждой амплитуда заменяется целевой и поле возвращается в плоскость ДОЭ.

    Отличие от параллельного варианта (`gerchberg_saxton`): поля плоскостей не суммируются,
    а применяются одно за другим, поэтому последняя плоскость в проходе получает преимущество.
    Это метод, на который ссылаются работы лаборатории. Одна итерация здесь равна одному
    полному обходу четырёх плоскостей, то есть вчетверо дороже итерации параллельного GS."""
    phase = random_phase(targets.shape[-2:], seed, targets.device)
    for _ in range(iterations):
        for plane in range(targets.shape[0]):
            chirp = chirps[plane:plane + 1]
            amplitude = targets[plane:plane + 1]
            field = propagate(phase, chirp)
            scale = field.abs().mean() / amplitude.mean().clamp_min(1e-9)
            constrained = torch.polar(amplitude * scale, field.angle())
            phase = back_propagate(constrained, chirp)[0].angle()
    return phase


def gradient_descent(targets: torch.Tensor, chirps: torch.Tensor, iterations: int = 3000, lr: float = 0.05,
                     seed: int = 0, init: torch.Tensor | None = None, mode: str = "amplitude") -> torch.Tensor:
    """Прямая оптимизация фазы по метрикам задачи через дифференцируемую модель Френеля."""
    phase = (random_phase(targets.shape[-2:], seed, targets.device) if init is None else init.clone()).float().requires_grad_()
    chirps = chirps.to(torch.complex64)
    targets = targets.float()
    optimizer = torch.optim.Adam([phase], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = image_loss(to_unit(reconstruct(phase, chirps, mode)), targets)
        loss.backward()
        optimizer.step()
        schedule.step()
    return phase.detach().double()
