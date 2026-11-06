"""Оценка предела многоплоскостного восстановления: ширина ядра связи плоскостей, оракул общих
низких частот и предел при одной плоскости на ДОЭ."""

import argparse
from pathlib import Path

import numpy as np

from holo.analysis import kernel_energy_radius, kernel_halfwidth, shared_lowpass_oracle, single_plane_ceiling
from holo.io import load_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", type=Path, nargs="+")
    args = parser.parse_args()
    print("полуширина ядра перехода между плоскостями, пиксели:")
    print(np.round(kernel_halfwidth(), 1))
    print("радиус 90 % энергии ядра от плоскости 1:", [kernel_energy_radius(0, j) for j in range(1, 4)])
    for folder in args.targets:
        names, images = load_targets(folder)
        for sigma in (2, 4, 8):
            m = shared_lowpass_oracle(images, sigma)
            print(f"{folder.name:7} оракул общих низких частот sigma={sigma}: CC {m['cc_mean']:.3f} SSIM {m['ssim_mean']:.3f}")
        m = single_plane_ceiling(images)
        print(f"{folder.name:7} одна плоскость на ДОЭ:                CC {m['cc_mean']:.3f} SSIM {m['ssim_mean']:.3f}")


if __name__ == "__main__":
    main()
