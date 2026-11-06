"""Прямой расчёт дифракции Френеля с одним преобразованием Фурье, как в условии задачи.

K(x, y) = exp(jkz) / (jλz) · exp(jπ(x² + y²) / λz) · FFT[H(u, v) · exp(jπ(u² + v²) / λz)]

Множители перед FFT по модулю постоянны, поэтому на восстановленную интенсивность
влияет только квадратичная фаза внутри FFT. Она и различает плоскости восстановления.
"""

from dataclasses import dataclass

import torch

WAVELENGTH = 532e-9
PIXEL = 10e-6
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
SIZE = 128


@dataclass(frozen=True)
class Setup:
    size: int = SIZE
    wavelength: float = WAVELENGTH
    pixel: float = PIXEL
    distances: tuple[float, ...] = DISTANCES

    def chirps(self, device=None) -> torch.Tensor:
        """Квадратичные фазовые множители exp(jπ(u² + v²)/λz) для каждой плоскости, форма (P, N, N)."""
        n = self.size
        coord = (torch.arange(n, dtype=torch.float64, device=device) - n / 2) * self.pixel
        r2 = coord[None, :] ** 2 + coord[:, None] ** 2
        z = torch.tensor(self.distances, dtype=torch.float64, device=device)[:, None, None]
        return torch.polar(torch.ones_like(r2 / z), torch.pi * r2 / (self.wavelength * z))

    def output_pixel(self, z: float) -> float:
        return self.wavelength * z / (self.size * self.pixel)


def propagate(phase: torch.Tensor, chirps: torch.Tensor) -> torch.Tensor:
    """Комплексное поле в плоскостях восстановления. phase (..., N, N) → (..., P, N, N), без постоянных множителей."""
    field = torch.polar(torch.ones_like(phase), phase).unsqueeze(-3) * chirps.to(phase.device)
    spectrum = torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho")
    return torch.fft.fftshift(spectrum, dim=(-2, -1))


def reconstruct(phase: torch.Tensor, chirps: torch.Tensor, mode: str = "amplitude") -> torch.Tensor:
    """Восстановленное изображение.

    По условию сохраняются модули значений поля, поэтому сравнение идёт по амплитуде |K|.
    Режим intensity оставлен для сравнения: от него зависит ИСС, но не КК.
    """
    field = propagate(phase, chirps).abs()
    return field if mode == "amplitude" else field**2


def intensity(phase: torch.Tensor, chirps: torch.Tensor) -> torch.Tensor:
    return reconstruct(phase, chirps, "intensity")


def phase_to_png(phase: torch.Tensor) -> torch.Tensor:
    """Фаза в 8-битный код: 0 соответствует −π, 255 соответствует +π."""
    wrapped = torch.remainder(phase + torch.pi, 2 * torch.pi) - torch.pi
    return torch.round((wrapped + torch.pi) / (2 * torch.pi) * 255).clamp(0, 255).to(torch.uint8)


def png_to_phase(code: torch.Tensor) -> torch.Tensor:
    return code.to(torch.float64) / 255 * 2 * torch.pi - torch.pi


def to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Нормировка восстановленного изображения по максимуму в диапазон 0..255."""
    peak = image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return torch.round(image / peak * 255).clamp(0, 255).to(torch.uint8)


def to_unit(image: torch.Tensor) -> torch.Tensor:
    """Дифференцируемая нормировка по максимуму в 0..1, как при сохранении в PNG."""
    return image / image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
