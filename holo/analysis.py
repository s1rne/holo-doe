"""Анализ предела многоплоскостного восстановления.

Поля в плоскостях связаны свёрткой: K_j = K_i ⊛ FFT(c_j / c_i), где c_z — квадратичный фазовый
множитель. Ядро — слабый чирп шириной в несколько пикселей, поэтому энергия в окне, заметно
большем ширины ядра, во всех плоскостях одинакова: низкие частоты интенсивности общие.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .metrics import report
from .optics import Setup


def kernel_halfwidth(setup: Setup = Setup()) -> np.ndarray:
    """Полуширина ядра перехода между плоскостями i и j в пикселях (по максимальной частоте чирпа)."""
    z = np.array(setup.distances)
    delta = np.abs(1 / z[:, None] - 1 / z[None, :]) / setup.wavelength
    r_max = setup.size / 2 * setup.pixel
    return r_max * delta * setup.size * setup.pixel


def kernel_energy_radius(i: int, j: int, setup: Setup = Setup(), fraction: float = 0.9) -> float:
    """Радиус (пиксели), в котором сосредоточена заданная доля энергии ядра FFT(c_j / c_i)."""
    chirps = setup.chirps()
    ratio = chirps[j] / chirps[i]
    kernel = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(ratio), norm="ortho")).abs().numpy() ** 2
    kernel /= kernel.sum()
    n = setup.size
    y, x = np.mgrid[:n, :n] - n // 2
    r = np.maximum(np.abs(x), np.abs(y)).ravel()
    order = np.argsort(r)
    return float(r[order][np.searchsorted(np.cumsum(kernel.ravel()[order]), fraction)])


def shared_lowpass_oracle(targets: np.ndarray, sigma: float) -> dict:
    """Оценка сверху при общих низких частотах: восстановление i = общий низкочастотный фон
    (среднее по целям) + собственные высокие частоты цели i. Метрики как при проверке."""
    images = targets.astype(np.float64) / 255
    low = np.stack([gaussian_filter(t, sigma, mode="wrap") for t in images])
    recon = np.clip(low.mean(0) + images - low, 0, None)
    recon = np.round(recon / recon.max(axis=(1, 2), keepdims=True) * 255).astype(np.uint8)
    return report(targets, recon)


def single_plane_ceiling(targets: np.ndarray, steps: int = 3000) -> dict:
    """Предел при одной плоскости на ДОЭ: четыре независимых ДОЭ, каждый под своё изображение."""
    from .models import PhaseField
    from .objective import Objective
    from .optimize import Schedule, fit
    from .evaluate import evaluate

    images = torch.from_numpy(targets).double() / 255
    setup = Setup()
    chirps = setup.chirps()
    recon = []
    for i in range(4):
        phase = fit(PhaseField(torch.zeros(1, setup.size, setup.size)), images[i:i + 1], chirps[i:i + 1],
                    Objective(peak_power=16), Schedule(steps, 0.5, quantize_from=0.5)).phase
        recon.append(evaluate(phase, targets[i:i + 1], Setup(distances=(setup.distances[i],)))[1][0])
    return report(targets, np.stack(recon))
