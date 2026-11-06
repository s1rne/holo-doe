"""Дифференцируемые версии метрик задачи. SSIM повторяет skimage.metrics.structural_similarity
с настройками по умолчанию: окно 7×7, K1 = 0.01, K2 = 0.03, несмещённая оценка ковариации."""

import torch
import torch.nn.functional as F


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(-2) - a.flatten(-2).mean(-1, keepdim=True)
    b = b.flatten(-2) - b.flatten(-2).mean(-1, keepdim=True)
    return (a * b).sum(-1) / ((a * a).sum(-1) * (b * b).sum(-1)).sqrt().clamp_min(1e-12)


def ssim(a: torch.Tensor, b: torch.Tensor, win: int = 7, data_range: float = 1.0) -> torch.Tensor:
    """a, b: (..., N, N) в диапазоне 0..data_range. Возвращает средний SSIM по изображению."""
    shape = a.shape
    a = a.reshape(-1, 1, *shape[-2:])
    b = b.reshape(-1, 1, *shape[-2:])
    kernel = torch.full((1, 1, win, win), 1.0 / win**2, dtype=a.dtype, device=a.device)
    cov_norm = win**2 / (win**2 - 1)
    mean = lambda x: F.conv2d(x, kernel)
    ua, ub = mean(a), mean(b)
    vaa = cov_norm * (mean(a * a) - ua * ua)
    vbb = cov_norm * (mean(b * b) - ub * ub)
    vab = cov_norm * (mean(a * b) - ua * ub)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * ua * ub + c1) * (2 * vab + c2)) / ((ua**2 + ub**2 + c1) * (vaa + vbb + c2))
    return s.mean(dim=(-3, -2, -1)).reshape(shape[:-2])


def image_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Сумма (1 − CC) и (1 − SSIM) по всем плоскостям."""
    return ((1 - pearson(recon, target)) + (1 - ssim(recon, target))).mean()
