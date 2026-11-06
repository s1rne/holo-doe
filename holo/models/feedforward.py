"""Прямой нейросетевой синтез ДОЭ: четыре целевых изображения → фаза за один проход сети.

Сеть повторяет структуру многоплоскостного Герчберга–Сэкстона, но каждый шаг делает обучаемым.
На шаге k U-Net в плоскостях восстановления смотрит на цели и на текущее поле в этих плоскостях
и предлагает комплексное поле g_z = T_z · u_z · exp(j∠K_z) (цель T_z сравнивается с амплитудой |K_z|). Поля обратно переносятся в плоскость
ДОЭ точной моделью Френеля и суммируются; второй U-Net поправляет сумму в плоскости ДОЭ, фаза
берётся как atan2 мнимой и действительной частей, поэтому разрыва фазы на ±π сеть не видит.
При u_z = 1 и нулевой поправке шаг совпадает с шагом GS, с этого состояния обучение и начинается.
"""

import torch
from torch import nn

from ..baselines import back_propagate
from ..optics import Setup, propagate
from .unet import UNet, coords as _coords

PLANES = len(Setup().distances)


def quantize(phase: torch.Tensor) -> torch.Tensor:
    """8-битная фаза как в PNG (0 → −π, 255 → +π), градиент проходит насквозь."""
    wrapped = torch.remainder(phase + torch.pi, 2 * torch.pi)
    code = torch.round(wrapped / (2 * torch.pi) * 255).clamp(0, 255)
    return phase + (code / 255 * 2 * torch.pi - torch.pi - phase).detach()


class FeedForwardDOE(nn.Module):
    """targets (B, 4, N, N) в 0..1 → фаза ДОЭ (B, N, N)."""

    def __init__(self, stages: int = 3, width: int = 32, depth: int = 4, doe_net: bool = True, setup: Setup = Setup()):
        super().__init__()
        self.register_buffer("chirps", setup.chirps().to(torch.complex64), persistent=False)
        # Вход сети в плоскостях: T, |K|, cos∠K, sin∠K по каждой плоскости и две координаты.
        self.image_nets = nn.ModuleList(UNet(4 * PLANES + 2, 2 * PLANES, width, depth) for _ in range(stages))
        self.doe_nets = nn.ModuleList(UNet(4, 2, width, depth) for _ in range(stages)) if doe_net else None

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = targets.shape
        amplitude = targets.clamp_min(0)
        coords = _coords(n, targets.device).expand(b, -1, -1, -1)
        field = torch.ones_like(amplitude, dtype=self.chirps.dtype)
        magnitude = torch.zeros_like(amplitude)
        for k, image_net in enumerate(self.image_nets):
            out = image_net(torch.cat([amplitude, magnitude, field.real, field.imag, coords], 1))
            u = torch.complex(1 + out[:, 0::2], out[:, 1::2])
            doe = back_propagate(amplitude * u * field, self.chirps).sum(1)
            doe = doe / doe.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
            if self.doe_nets is not None:
                fix = self.doe_nets[k](torch.cat([doe.real[:, None], doe.imag[:, None], coords], 1))
                doe = doe + torch.complex(fix[:, 0], fix[:, 1])
            phase = torch.atan2(doe.imag, doe.real)
            recon = propagate(phase, self.chirps)
            magnitude = recon.abs()
            magnitude = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            field = recon / recon.abs().clamp_min(1e-8)
        return phase


def save_model(model: FeedForwardDOE, config: dict, path) -> None:
    torch.save({"config": config, "state": model.state_dict()}, path)


def load_model(path, device="cpu") -> tuple[FeedForwardDOE, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = FeedForwardDOE(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state"])
    return model.eval(), checkpoint["config"]
