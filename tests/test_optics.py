"""Проверка модели Френеля независимыми способами.

Ошибка в знаке, в порядке fftshift или в выборе FFT против IFFT не ломает расчёт
формально, но разворачивает или сдвигает восстановленное изображение. Жюри считает
метрики своей реализацией, поэтому расхождение стоит всех баллов.
"""

import numpy as np
import torch

from holo.optics import Setup, intensity, phase_to_png, png_to_phase, propagate

SETUP = Setup()


def direct_fresnel(field: np.ndarray, z: float, setup: Setup = SETUP) -> np.ndarray:
    """Та же дифракция, посчитанная двойной суммой по определению, без БПФ."""
    n = setup.size
    coord = (np.arange(n) - n / 2) * setup.pixel
    out_coord = coord * setup.wavelength * z / (n * setup.pixel**2)
    k = 2 * np.pi / setup.wavelength
    u, v = np.meshgrid(coord, coord, indexing="xy")
    source = field * np.exp(1j * np.pi * (u**2 + v**2) / (setup.wavelength * z))
    result = np.zeros((n, n), dtype=complex)
    for i, y in enumerate(out_coord):
        for j, x in enumerate(out_coord):
            kernel = np.exp(-2j * np.pi * (x * u + y * v) / (setup.wavelength * z))
            result[i, j] = np.exp(1j * k * z) / (1j * setup.wavelength * z) * np.sum(source * kernel)
    return result


def test_fft_matches_direct_summation():
    """Реализация через FFT совпадает с прямым суммированием с точностью до постоянного множителя."""
    torch.manual_seed(0)
    small = Setup(size=16)
    phase = (torch.rand(16, 16, dtype=torch.float64) * 2 - 1) * torch.pi
    fast = propagate(phase, small.chirps())[0].numpy()
    slow = direct_fresnel(np.exp(1j * phase.numpy()), small.distances[0], small)

    scale = np.vdot(fast, slow) / np.vdot(fast, fast)
    assert np.allclose(fast * scale, slow, rtol=1e-8, atol=1e-8 * np.abs(slow).max())


def test_lens_phase_focuses_to_center():
    """Фаза тонкой линзы с фокусом на первой плоскости собирает свет в её центр."""
    n = SETUP.size
    coord = (torch.arange(n, dtype=torch.float64) - n / 2) * SETUP.pixel
    r2 = coord[None, :] ** 2 + coord[:, None] ** 2
    z = SETUP.distances[0]
    phase = -torch.pi * r2 / (SETUP.wavelength * z)

    image = intensity(phase, SETUP.chirps())[0]
    peak = torch.unravel_index(image.argmax(), image.shape)

    assert (peak[0].item(), peak[1].item()) == (n // 2, n // 2)
    assert image[n // 2, n // 2] / image.sum() > 0.5


def test_shift_of_doe_does_not_shift_image():
    """Сдвиг ДОЭ не сдвигает картинку: признак того, что квадратичная фаза учтена, а не потеряна."""
    torch.manual_seed(1)
    phase = (torch.rand(SETUP.size, SETUP.size, dtype=torch.float64) * 2 - 1) * torch.pi
    chirps = SETUP.chirps()

    base = intensity(phase, chirps)[0]
    rolled = intensity(torch.roll(phase, shifts=(3, 0), dims=(0, 1)), chirps)[0]

    assert not torch.allclose(base, rolled)


def test_png_roundtrip_is_within_one_step():
    """Фаза периодична, поэтому разность сравнивается по кратчайшей дуге."""
    phase = torch.linspace(-torch.pi, torch.pi, 128, dtype=torch.float64).repeat(128, 1)
    restored = png_to_phase(phase_to_png(phase))
    error = torch.remainder(restored - phase + torch.pi, 2 * torch.pi) - torch.pi
    assert error.abs().max() <= torch.pi / 255


def test_png_codes_match_specification():
    """0 соответствует −π, 255 соответствует +π."""
    value = lambda x: phase_to_png(torch.tensor([x], dtype=torch.float64)).item()
    assert value(-torch.pi) == 0
    assert value(torch.pi - 1e-9) == 255
    assert value(0.0) == 128
    # Ровно +π физически совпадает с −π и кодируется нулём.
    assert value(torch.pi) == 0


def test_output_pixel_size_matches_task_scale():
    """При z = N·δ²/λ масштаб восстановления равен масштабу ДОЭ."""
    z = SETUP.size * SETUP.pixel**2 / SETUP.wavelength
    assert abs(SETUP.output_pixel(z) - SETUP.pixel) < 1e-12
    assert abs(z - 24.06e-3) < 0.1e-3
