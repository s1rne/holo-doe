"""Проверка добавленных слагаемых потери: штраф за спекл и штраф за недобор средней яркости.

Оба введены против вырожденного решения, которое набирает SSIM за счёт того, что фон после
нормировки по максимуму округляется в ноль. Тесты фиксируют, что штрафы ловят именно это.
"""

import torch

from holo.losses import speckle_excess
from holo.objective import Objective
from holo.optics import Setup

SETUP = Setup()


def lens_phase() -> torch.Tensor:
    """Фаза тонкой линзы: собирает весь свет в один пиксель первой плоскости. Это предельный
    случай вырожденного решения — максимум огромен, фон после нормировки нулевой."""
    n = SETUP.size
    coord = (torch.arange(n, dtype=torch.float64) - n / 2) * SETUP.pixel
    r2 = coord[None, :] ** 2 + coord[:, None] ** 2
    return -torch.pi * r2 / (SETUP.wavelength * SETUP.distances[0])


def test_speckle_excess_is_zero_on_exact_match():
    target = torch.rand(2, 4, 32, 32, dtype=torch.float64)
    assert torch.allclose(speckle_excess(target, target), torch.zeros(2, 4, dtype=torch.float64))


def test_speckle_excess_grows_with_noise():
    torch.manual_seed(0)
    target = torch.zeros(1, 1, 32, 32, dtype=torch.float64)
    target[..., 8:24, 8:24] = 1.0
    weak = speckle_excess(target + 0.02 * torch.randn_like(target), target)
    strong = speckle_excess(target + 0.2 * torch.randn_like(target), target)
    assert 0 < float(weak) < float(strong)


def test_mean_penalty_separates_degenerate_solution():
    """Штраф за среднюю яркость должен ухудшать линзовую фазу сильнее, чем случайную."""
    torch.manual_seed(0)
    chirps = SETUP.chirps()
    targets = torch.rand(4, SETUP.size, SETUP.size, dtype=torch.float64) * 0.5 + 0.25
    random = (torch.rand(SETUP.size, SETUP.size, dtype=torch.float64) * 2 - 1) * torch.pi
    plain, penalized = Objective(), Objective(mean_weight=1.0)

    lens_cost = float(penalized(lens_phase(), chirps, targets) - plain(lens_phase(), chirps, targets))
    random_cost = float(penalized(random, chirps, targets) - plain(random, chirps, targets))

    # Случайная фаза тоже даёт пик выше среднего и тоже получает штраф, но вчетверо меньший.
    assert lens_cost > 4 * random_cost
    assert 0 < random_cost < 1


def test_worst_weight_blends_mean_and_softmin():
    """При worst_weight = 0 мягкий минимум не влияет: потеря совпадает с обычным средним."""
    torch.manual_seed(0)
    chirps = SETUP.chirps()
    targets = torch.rand(4, SETUP.size, SETUP.size, dtype=torch.float64)
    phase = (torch.rand(SETUP.size, SETUP.size, dtype=torch.float64) * 2 - 1) * torch.pi

    mean_only = Objective()(phase, chirps, targets)
    blended = Objective(softmin=0.05, worst_weight=0.0)(phase, chirps, targets)
    worst = Objective(softmin=0.05, worst_weight=1.0)(phase, chirps, targets)

    assert torch.allclose(mean_only, blended, atol=1e-9)
    assert float(worst) > float(mean_only)
