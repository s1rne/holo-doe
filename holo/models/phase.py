import torch
from torch import nn


class PhaseField(nn.Module):
    """Фаза как свободный параметр: обычный градиентный спуск. init (B, N, N)."""

    def __init__(self, init: torch.Tensor):
        super().__init__()
        self.phase = nn.Parameter(init.float().clone())

    def forward(self) -> torch.Tensor:
        return self.phase


class ComplexField(nn.Module):
    """Фаза как угол свободного комплексного поля: перепараметризация без границ, из того же
    семейства, что выход сетей (atan2 по двум каналам)."""

    def __init__(self, init: torch.Tensor):
        super().__init__()
        self.xy = nn.Parameter(torch.stack([init.cos(), init.sin()], 1).float())

    def forward(self) -> torch.Tensor:
        return to_phase(self.xy)


def to_phase(xy: torch.Tensor) -> torch.Tensor:
    """Два канала (cos, sin) в фазу через atan2: выход сети не упирается в границы ±π."""
    return torch.atan2(xy[:, 1], xy[:, 0])
