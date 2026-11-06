"""Кривая «качество — время»: нейросетевой метод против классических итеративных.

Все методы считают одну и ту же задачу на одном устройстве, оцениваются одинаково
(`holo.evaluate.evaluate`: фаза в 8-битный PNG, восстановление из него по амплитуде |K|,
нормировка по максимуму, округление до 8 бит, КК и SSIM через skimage).

Колонка «яркость» — отношение средней яркости восстановления к средней яркости цели после
нормировки по максимуму. Она показывает вырожденное решение: когда часть энергии уходит в один
яркий пик, фон округляется в ноль, SSIM растёт, а яркость падает заметно ниже единицы.

    PYTHONPATH=. uv run python scripts/bench.py data/targets/photos --checkpoint runs/neural/model.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from holo.baselines import gerchberg_saxton, gradient_descent, ping_pong, random_phase
from holo.evaluate import evaluate, save
from holo.io import load_targets
from holo.models import FourierMLP, PhaseField, Siren
from holo.models.learned import load_model
from holo.neural import synthesize
from holo.objective import Objective
from holo.optics import Setup
from holo.optimize import Schedule, device, fit

ROOT = Path(__file__).resolve().parent.parent
EXACT = Objective()


def gd(steps: int, lr: float = 0.3, objective: Objective = EXACT):
    return lambda t, c: fit(PhaseField(random_phase((128, 128), 0)[None]), t, c, objective,
                            Schedule(steps, lr, quantize_from=0.5)).phase


def classical(name: str, budgets: dict) -> dict:
    """Классические методы: параллельный GS, последовательный ping-pong и градиентный спуск."""
    methods = {}
    for n in budgets["gs"]:
        methods[f"GS {n}"] = (lambda n: lambda t, c: gerchberg_saxton(t, c, n))(n)
    for n in budgets["pp"]:
        methods[f"ping-pong {n}"] = (lambda n: lambda t, c: ping_pong(t, c, n))(n)
    for n in budgets["gd"]:
        methods[f"GD {n}"] = gd(n)
    for n in budgets["gsgd"]:
        methods[f"GS 50 + GD {n}"] = (lambda n: lambda t, c: fit(
            PhaseField(gerchberg_saxton(t, c, 50)[None]), t, c, EXACT, Schedule(n, 0.3, quantize_from=0.5)).phase)(n)
    return methods


def build_methods(args, model) -> dict:
    budgets = {"gs": args.gs, "pp": args.pp, "gd": args.gd, "gsgd": args.gsgd}
    methods = classical("classical", budgets) if not args.only_neural else {}
    if args.implicit:
        methods["SIREN 3k"] = lambda t, c: fit(Siren(), t, c, EXACT, Schedule(3000, 1e-4, quantize_from=0.5)).phase
        methods["fourier-MLP 3k"] = lambda t, c: fit(FourierMLP(sigma=64), t, c, EXACT,
                                                     Schedule(3000, 1e-3, quantize_from=0.5)).phase
    if model is not None:
        for n in args.neural:
            methods[f"сеть {n} шагов"] = (lambda n: lambda t, c: synthesize(model, t, c, "fast", n, 1).phase)(n)
        for n in args.seeds:
            methods[f"сеть {model.steps} шагов × {n} стартов"] = (
                lambda n: lambda t, c: synthesize(model, t, c, "neural", None, n).phase)(n)
        for n in args.neural_gd:
            methods[f"сеть + GD {n}"] = (lambda n: lambda t, c: synthesize(model, t, c, "quality", None, 4, n).phase)(n)
    return methods


def brightness(recon: np.ndarray, images: np.ndarray) -> float:
    return float(np.mean(recon.mean(axis=(-2, -1)) / np.maximum(images.mean(axis=(-2, -1)), 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "bench")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--gs", type=int, nargs="*", default=[50, 200, 1000])
    parser.add_argument("--pp", type=int, nargs="*", default=[25, 100, 400])
    parser.add_argument("--gd", type=int, nargs="*", default=[50, 200, 1000, 3000])
    parser.add_argument("--gsgd", type=int, nargs="*", default=[200, 1000])
    parser.add_argument("--neural", type=int, nargs="*", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="*", default=[4])
    parser.add_argument("--neural-gd", type=int, nargs="*", default=[200, 500])
    parser.add_argument("--implicit", action="store_true", help="добавить SIREN и fourier-MLP")
    parser.add_argument("--only-neural", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    dev = device()
    chirps = Setup().chirps().to(torch.complex64).to(dev)
    model = load_model(args.checkpoint, dev) if args.checkpoint else None
    methods = build_methods(args, model)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in args.targets:
        names, images = load_targets(folder)
        targets = torch.from_numpy(images).float().to(dev) / 255
        for method, run in methods.items():
            torch.manual_seed(0)
            start = time.perf_counter()
            phase = run(targets, chirps)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            code, recon, metrics = evaluate(phase.reshape(128, 128).cpu().double(), images)
            row = {"set": folder.name, "method": method, "cc": metrics["cc_mean"], "ssim": metrics["ssim_mean"],
                   "cc_min": min(metrics["cc"]), "ssim_min": min(metrics["ssim"]),
                   "brightness": brightness(recon, images), "seconds": seconds}
            rows.append(row)
            if args.save_images:
                slug = method.replace(" ", "_").replace("×", "x").replace("+", "_")
                save(args.out / args.tag / folder.name / slug, names, code, recon, metrics,
                     {"method": method, "seconds": seconds})
            print(f"| {row['set']} | {method} | {row['cc']:.3f} | {row['ssim']:.3f} | {row['cc'] + row['ssim']:.3f} | "
                  f"{row['cc_min']:.3f} | {row['ssim_min']:.3f} | {row['brightness']:.2f} | {seconds:.2f} |", flush=True)
    with (args.out / f"{args.tag}.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
