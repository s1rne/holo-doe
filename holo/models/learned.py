"""Обучаемый оптимизатор фазы ДОЭ: развёрнутая схема, где каждый шаг делает свёрточная сеть.

Шаг k получает локальные признаки текущего решения — фазу ДОЭ, градиент целевой функции по фазе,
поправку многоплоскостного Герчберга–Сэкстона, цели и остаток «цель минус амплитуда поля» — и
предлагает приращение фазы. Физика (модель Френеля из условия) входит в каждый шаг: признаки
считаются через `propagate`/`back_propagate`, поэтому сеть не заучивает оптику, а учится править
решение. Начальная фаза тоже даёт сеть: U-Net по четырём целям выбирает бассейн, из которого
развёрнутые шаги сходятся лучше, чем из случайной фазы.

Веса свёрток общие для всех шагов, различие шагов задают обучаемый сдвиг на первом слое и
обучаемый масштаб приращения. Выходная свёртка инициализирована нулём: необученная сеть фазу
не двигает и совпадает со своей начальной фазой.
"""

import torch
from torch import nn

from ..baselines import back_propagate, random_phase
from ..objective import Objective
from ..optics import Setup, propagate
from .unet import UNet, coords

PLANES = len(Setup().distances)


def conv(cin: int, cout: int, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, padding_mode="circular")


class StepNet(nn.Module):
    """Свёрточный блок одного шага. Дилатации 1, 2, 4, 8 дают радиус 15 пикселей — порядок
    полуширины ядра связи между крайними плоскостями (8,3 пикселя, см. docs/per_instance.md),
    поэтому сеть видит тот масштаб, на котором плоскости конфликтуют."""

    def __init__(self, cin: int, width: int = 64, dilations: tuple[int, ...] = (1, 2, 4, 8, 1), steps: int = 32):
        super().__init__()
        self.first = conv(cin, width, dilations[0])
        self.step_bias = nn.Parameter(torch.zeros(steps, width))
        body: list[nn.Module] = []
        for dilation in dilations[1:]:
            body += [nn.GELU(), conv(width, width, dilation)]
        self.body = nn.Sequential(*body)
        self.out = conv(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        h = self.first(x) + self.step_bias[step][None, :, None, None]
        return self.out(self.body(h))[:, 0]


class LearnedOptimizer(nn.Module):
    FEATURES = 7

    def __init__(self, steps: int = 32, width: int = 64, init_net: bool = True, init_width: int = 24,
                 init_depth: int = 3, dilations: tuple[int, ...] = (1, 2, 4, 8, 1)):
        super().__init__()
        self.net = StepNet(self.FEATURES + 2 * PLANES, width, dilations, steps)
        self.init = UNet(PLANES + 2, 2, init_width, init_depth) if init_net else None
        self.steps = steps
        self.gain = nn.Parameter(torch.full((steps,), 1.0))

    @property
    def config(self) -> dict:
        return {"steps": self.steps, "width": self.net.first.out_channels,
                "init_net": self.init is not None,
                "init_width": self.init.inp.body[0].out_channels if self.init else 0,
                "init_depth": len(self.init.down) if self.init else 0}

    def start(self, targets: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Начальная фаза (B, N, N): случайная фаза, повёрнутая поправкой сети по целям.

        Поправка складывается с единичным вектором exp(jφ) и фаза берётся как atan2, поэтому
        при нулевом выходе сети старт совпадает со случайной фазой, а при большом выходе сеть
        задаёт старт целиком. Разрыва на ±π сеть не видит."""
        batch, _, n, _ = targets.shape
        base = torch.stack([random_phase((n, n), seed + b) for b in range(batch)]).float().to(targets.device)
        if self.init is None:
            return base
        grid = coords(n, targets.device).expand(batch, -1, -1, -1)
        out = self.init(torch.cat([targets, grid], 1))
        return torch.atan2(base.sin() + out[:, 1], base.cos() + out[:, 0])

    def features(self, phase, targets, chirps, objective):
        with torch.enable_grad():
            probe = phase.detach().requires_grad_()
            grad = torch.autograd.grad(objective(probe, chirps, targets).sum(), probe)[0]
        grad = torch.nan_to_num(grad / grad.flatten(-2).std(-1)[..., None, None].clamp_min(1e-9)).clamp(-5, 5)
        field = propagate(phase, chirps)
        amplitude = targets
        scale = field.abs().mean((-2, -1), keepdim=True) / amplitude.mean((-2, -1), keepdim=True).clamp_min(1e-3)
        back = back_propagate(torch.polar(amplitude * scale, field.angle()), chirps).sum(-3)
        turn = back * torch.polar(torch.ones_like(phase), -phase)
        magnitude = back.abs().clamp_min(1e-6)
        weight = magnitude / magnitude.mean((-2, -1), keepdim=True)
        unit = field.abs() / field.abs().amax((-2, -1), keepdim=True).clamp_min(1e-9)
        features = torch.stack([phase.cos(), phase.sin(), grad, turn.real / magnitude, turn.imag / magnitude,
                                weight, field.abs().mean(-3)], 1)
        features = torch.cat([features, targets, targets - unit], 1)
        return torch.nan_to_num(features).clamp(-10, 10)

    def step(self, phase: torch.Tensor, targets: torch.Tensor, chirps: torch.Tensor,
             objective: Objective, index: int) -> torch.Tensor:
        """Одно приращение фазы. index выбирает сдвиг и масштаб шага; шаги за пределами обучения
        повторяют последний."""
        index = min(index, self.steps - 1)
        x = self.features(phase, targets, chirps, objective)
        return phase + self.gain[index] * torch.tanh(self.net(x, index))

    def forward(self, phase: torch.Tensor, targets: torch.Tensor, chirps: torch.Tensor,
                objective: Objective = Objective(quantize=False), steps: int | None = None) -> list[torch.Tensor]:
        """phase (B, N, N), targets (B, P, N, N). Возвращает фазы после каждого шага."""
        history = []
        for k in range(steps if steps is not None else self.steps):
            phase = self.step(phase, targets, chirps, objective, k)
            history.append(phase)
        return history


def save_model(model: LearnedOptimizer, path) -> None:
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load_model(path, device="cpu") -> LearnedOptimizer:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config") or {"steps": checkpoint["steps"]}
    model = LearnedOptimizer(**config).to(device)
    model.load_state_dict(checkpoint["state"])
    return model.eval()
