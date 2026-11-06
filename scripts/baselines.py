"""Прогон эталонных методов на тестовых наборах: качество и время."""

import argparse
import time
from pathlib import Path

import torch

from holo.baselines import gerchberg_saxton, gradient_descent
from holo.evaluate import evaluate, save
from holo.io import load_targets
from holo.optics import Setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/baselines"))
    args = parser.parse_args()
    chirps = Setup().chirps()
    for folder in args.targets:
        names, images = load_targets(folder)
        targets = torch.from_numpy(images).double() / 255
        runs = {
            "gs200": lambda: gerchberg_saxton(targets, chirps, 200),
            "gd3000": lambda: gradient_descent(targets, chirps, 3000),
            "gs50+gd3000": lambda: gradient_descent(targets, chirps, 3000, init=gerchberg_saxton(targets, chirps, 50)),
        }
        for method, run in runs.items():
            start = time.perf_counter()
            phase = run()
            seconds = time.perf_counter() - start
            code, recon, metrics = evaluate(phase, images)
            save(args.out / folder.name / method, names, code, recon, metrics, {"method": method, "seconds": seconds})
            print(f"{folder.name:8} {method:12} CC {metrics['cc_mean']:.4f}  SSIM {metrics['ssim_mean']:.4f}  {seconds:6.1f} с  "
                  f"CC {[round(x, 3) for x in metrics['cc']]}  SSIM {[round(x, 3) for x in metrics['ssim']]}")


if __name__ == "__main__":
    main()
