"""Сравнение per-instance методов на тестовых наборах. Оценка только через holo.evaluate.evaluate
(восстановление из сохранённого 8-битного PNG); дополнительно метрики при амплитудном определении."""

import argparse
import json
import time
from pathlib import Path

import torch

from holo.baselines import gerchberg_saxton, gradient_descent, random_phase
from holo.evaluate import evaluate, save
from holo.io import load_targets
from holo.models import ComplexField, DeepPrior, FourierMLP, PhaseField, Siren
from holo.models.learned import LearnedOptimizer
from holo.objective import Objective
from holo.optics import Setup
from holo.optimize import Schedule, device, fit
from holo.pretrain import smooth_target_fit

EXACT = Objective(peak_power=16)


def restarts(count: int, seed: int = 0) -> torch.Tensor:
    return torch.stack([random_phase((128, 128), seed + i) for i in range(count)])


def gd(steps: int, objective: Objective = EXACT, count: int = 1, lr: float = 0.3, seconds: float | None = None):
    return lambda t, c: fit(PhaseField(restarts(count)), t, c, objective, Schedule(steps, lr, quantize_from=0.5, seconds=seconds)).phase


def learned(checkpoint: Path, finetune: int):
    def run(t, c):
        state = torch.load(checkpoint, map_location="cpu")
        model = LearnedOptimizer(state["steps"])
        model.load_state_dict(state["state"])
        model = model.to(device()).eval()
        with torch.no_grad():
            phase = model(restarts(1).float().to(device()), t.float()[None].to(device()), c.to(torch.complex64).to(device()),
                          Objective(quantize=False, peak_power=16))[-1].cpu()
        if finetune:
            phase = fit(PhaseField(phase), t, c, EXACT, Schedule(finetune, 0.5, quantize_from=0.5)).phase
        return phase.reshape(128, 128)
    return run


def smooth(scale: int = 4, pre: int = 3000, steps: int = 3000, lr: float = 0.1, objective: Objective = EXACT):
    def run(t, c):
        init = smooth_target_fit(t, c, scale, pre)[None]
        return fit(PhaseField(init), t, c, objective, Schedule(steps, lr, quantize_from=0.5)).phase
    return run


METHODS = {
    "gs200 (базовый)": lambda t, c: gerchberg_saxton(t, c, 200),
    "gd3000 (базовый)": lambda t, c: gradient_descent(t, c, 3000),
    "gs50+gd3000 (базовый)": lambda t, c: gradient_descent(t, c, 3000, init=gerchberg_saxton(t, c, 50)),
    "gd-exact 500": gd(500),
    "gd-exact 3k": gd(3000),
    "gd-exact 10k": gd(10000),
    "gd-exact 10k ×8 рестартов": gd(10000, count=8),
    "gs50 + gd-exact 3k": lambda t, c: fit(PhaseField(gerchberg_saxton(t, c, 50)[None]), t, c, EXACT,
                                           Schedule(3000, 0.3, quantize_from=0.5)).phase,
    "gd-exact 3k, мягкий минимум": gd(3000, Objective(peak_power=16, softmin=0.05)),
    "gd-exact 3k, без квантования": gd(3000, Objective(peak_power=16, quantize=False)),
    "gd-exact 3k, без мягкого пика": gd(3000, Objective()),
    "gd-exact 3k, затемнение 2": gd(3000, Objective(peak_power=16, dim_weight=2.0)),
    "gd-exact 3k, |K| и |K|²": gd(3000, Objective(peak_power=16, modes=("amplitude", "intensity"))),
    "гладкая фаза + gd-exact 1k": smooth(steps=1000),
    "гладкая фаза + gd-exact 3k": smooth(),
    "гладкая фаза + gd-exact 3k, затемнение 2": smooth(objective=Objective(peak_power=16, dim_weight=2.0)),
    "complex-field 3k": lambda t, c: fit(ComplexField(restarts(1)), t, c, EXACT, Schedule(3000, 0.2, quantize_from=0.5)).phase,
    "dip-noise 3k": lambda t, c: fit(DeepPrior(torch.randn(1, 32, 128, 128, generator=torch.Generator().manual_seed(0))),
                                     t, c, EXACT, Schedule(3000, 1e-3, quantize_from=0.5)).phase,
    "siren 3k": lambda t, c: fit(Siren(), t, c, EXACT, Schedule(3000, 1e-4, quantize_from=0.5)).phase,
    "fourier-mlp 3k": lambda t, c: fit(FourierMLP(sigma=64), t, c, EXACT, Schedule(3000, 1e-3, quantize_from=0.5)).phase,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", type=Path, nargs="+")
    parser.add_argument("--methods", nargs="*", default=None, help="подстроки имён методов")
    parser.add_argument("--learned", type=Path, default=None, help="чекпойнт обучаемого оптимизатора")
    parser.add_argument("--out", type=Path, default=Path("outputs/per_instance"))
    args = parser.parse_args()
    methods = dict(METHODS)
    if args.learned:
        methods["learned-opt 16 шагов"] = learned(args.learned, 0)
        methods["learned-opt 16 + gd-exact 500"] = learned(args.learned, 500)
    if args.methods:
        methods = {k: v for k, v in methods.items() if any(m in k for m in args.methods)}
    chirps = Setup().chirps()
    rows = []
    for folder in args.targets:
        names, images = load_targets(folder)
        targets = torch.from_numpy(images).double() / 255
        for method, run in methods.items():
            torch.manual_seed(0)
            start = time.perf_counter()
            phase = run(targets, chirps)
            seconds = time.perf_counter() - start
            code, recon, metrics = evaluate(phase, images)
            other = evaluate(phase, images, mode="intensity")[2]
            slug = method.split(" (")[0].replace(" ", "_").replace("×", "x").replace("+", "_")
            save(args.out / folder.name / slug, names, code, recon, metrics, {"method": method, "seconds": seconds})
            row = {"set": folder.name, "method": method, "cc": metrics["cc_mean"], "ssim": metrics["ssim_mean"],
                   "cc_min": min(metrics["cc"]), "ssim_min": min(metrics["ssim"]),
                   "cc_int": other["cc_mean"], "ssim_int": other["ssim_mean"], "seconds": seconds}
            rows.append(row)
            print(f"| {folder.name} | {method} | {row['cc']:.3f} | {row['ssim']:.3f} | {row['cc'] + row['ssim']:.3f} | "
                  f"{row['cc_min']:.3f} | {row['ssim_min']:.3f} | {row['cc_int']:.3f} | {row['ssim_int']:.3f} | {seconds:.1f} |", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
