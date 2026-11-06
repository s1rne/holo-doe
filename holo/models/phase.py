import torch
from torch import nn


class PhaseField(nn.Module):
    """Фаза как свободный параметр: обычный градиентный спуск. init (B, N, N)."""

    def __init__(self, init: torch.Tensor):
        super().__init__()
        self.phase = nn.Parameter(init.float().clone())

    def forward(self) -> torch.Tensor:
        return self.phase


def to_phase(xy: torch.Tensor) -> torch.Tensor:
    """Два канала (cos, sin) в фазу через atan2: выход сети не упирается в границы ±π."""
    return torch.atan2(xy[:, 1], xy[:, 0])
