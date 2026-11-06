"""Обучаемый оптимизатор (развёрнутый GS/градиентный спуск): маленькая CNN по локальным признакам
текущей фазы, градиента потери и GS-поправки предлагает шаг фазы. Обучается на многих сценах,
затем применяется к новой сцене за фиксированное число шагов."""

import torch
from torch import nn

from ..baselines import back_propagate
from ..objective import Objective
from ..optics import propagate


def conv(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular")


class LearnedOptimizer(nn.Module):
    FEATURES = 7

    def __init__(self, steps: int = 16, width: int = 48, depth: int = 4):
        super().__init__()
        layers, cin = [], self.FEATURES + 4
        for _ in range(depth):
            layers += [conv(cin, width), nn.GELU()]
            cin = width
        self.net = nn.Sequential(*layers, conv(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.steps = steps
        self.gain = nn.Parameter(torch.full((steps,), 1.0))

    def features(self, phase, targets, chirps, objective):
        with torch.enable_grad():
            probe = phase.detach().requires_grad_()
            grad = torch.autograd.grad(objective(probe, chirps, targets).sum(), probe)[0]
        grad = torch.nan_to_num(grad / grad.flatten(-2).std(-1)[..., None, None].clamp_min(1e-9)).clamp(-5, 5)
        field = propagate(phase, chirps)
        amplitude = targets
        scale = field.abs().mean((-2, -1), keepdim=True) / amplitude.mean((-2, -1), keepdim=True).clamp_min(1e-6)
        back = back_propagate(torch.polar(amplitude * scale, field.angle()), chirps).sum(-3)
        turn = back * torch.polar(torch.ones_like(phase), -phase)
        weight = back.abs() / back.abs().mean((-2, -1), keepdim=True)
        return torch.stack([phase.cos(), phase.sin(), grad, turn.real / back.abs().clamp_min(1e-9),
                            turn.imag / back.abs().clamp_min(1e-9), weight, field.abs().pow(2).mean(-3)], 1)

    def forward(self, phase: torch.Tensor, targets: torch.Tensor, chirps: torch.Tensor,
                objective: Objective = Objective(quantize=False), steps: int | None = None) -> list[torch.Tensor]:
        """phase (B, N, N), targets (B, P, N, N). Возвращает фазы после каждого шага."""
        history = []
        for k in range(steps or self.steps):
            x = torch.cat([self.features(phase, targets, chirps, objective), targets], 1)
            phase = phase + self.gain[min(k, self.steps - 1)] * torch.tanh(self.net(x)[:, 0])
            history.append(phase)
        return history
