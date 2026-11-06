"""Метрики из условия: коэффициент корреляции Пирсона и индекс структурного сходства (SSIM)."""

import numpy as np
from skimage.metrics import structural_similarity


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else 0.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=255))


def report(targets: np.ndarray, recon: np.ndarray) -> dict:
    cc = [pearson(t, r) for t, r in zip(targets, recon)]
    ss = [ssim(t, r) for t, r in zip(targets, recon)]
    return {"cc": cc, "ssim": ss, "cc_mean": float(np.mean(cc)), "ssim_mean": float(np.mean(ss))}
