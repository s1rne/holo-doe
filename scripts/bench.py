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

from holo.baselines import gerchberg_saxton, ping_pong, random_phase
from holo.evaluate import evaluate, save
from holo.io import load_targets
from holo.models import FourierMLP, PhaseField, Siren
from holo.models.learned import load_model
from holo.neural import synthesize
from holo.objective import Objective
from holo.optics import Setup
from holo.optimize import Schedule, device, fit, synchronize

ROOT = Path(__file__).resolve().parent.parent
EXACT = Objective()


def gd(steps: int, lr: float = 0.3, objective: Objective = EXACT):
    return lambda t, c: fit(PhaseField(random_phase((128, 128), 0)[None]), t, c, objective,
                            Schedule(steps, lr, quantize_from=0.5)).phase


def classical(budgets: dict) -> dict:
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


def build_methods(args, model, forward_net) -> dict:
    budgets = {"gs": args.gs, "pp": args.pp, "gd": args.gd, "gsgd": args.gsgd}
    methods = {} if args.only_neural else classical(budgets)
    if args.implicit:
        methods["SIREN 3k"] = lambda t, c: fit(Siren(), t, c, EXACT, Schedule(3000, 1e-4, quantize_from=0.5)).phase
        methods["fourier-MLP 3k"] = lambda t, c: fit(FourierMLP(sigma=64), t, c, EXACT,
                                                     Schedule(3000, 1e-3, quantize_from=0.5)).phase
    if forward_net is not None:
        methods["ФФ-сеть, 1 проход"] = lambda t, c: forward_net(t[None].float())[0].detach()
    if model is None:
        return methods
    name = args.name or ("ФФ-старт" if forward_net is not None else "свой старт")
    run = lambda **kw: lambda t, c: synthesize(model, t, c, forward_net=forward_net, **kw).phase
    for n in args.neural:
        methods[f"сеть ({name}) {n} шагов"] = run(mode="fast", steps=n, seeds=1)
    for n in args.seeds:
        methods[f"сеть ({name}) {model.steps} шагов × {n} стартов"] = run(mode="neural", seeds=n)
    for n in args.neural_gd:
        methods[f"сеть ({name}) + GD {n}"] = run(mode="quality", seeds=4, refine=n)
    if args.ablate:
        methods[f"сеть ({name}) {model.steps} шагов, старт случайный"] = \
            lambda t, c: synthesize(model, t, c, "fast", None, 1, use_init=False).phase
        methods[f"сеть ({name}), только начальная фаза"] = run(mode="fast", steps=0, seeds=1)
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
    parser.add_argument("--ablate", action="store_true", help="добавить разборы: старт без сети, только начальная фаза")
    parser.add_argument("--ff-checkpoint", type=Path, default=None, help="чекпойнт прямой сети (ветка exp/feedforward)")
    parser.add_argument("--ff-init", action="store_true", help="начальную фазу оптимизатора даёт прямая сеть")
    parser.add_argument("--name", default=None, help="подпись варианта сети в именах методов")
    parser.add_argument("--only-neural", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    dev = device()
    chirps = Setup().chirps().to(torch.complex64).to(dev)
    model = load_model(args.checkpoint, dev) if args.checkpoint else None
    forward_net = None
    if args.ff_checkpoint:
        from holo.models.feedforward import load_model as load_forward
        forward_net = load_forward(args.ff_checkpoint, dev)[0]
    methods = build_methods(args, model, forward_net if args.ff_init else None)
    if args.ff_checkpoint and not args.ff_init:
        methods["ФФ-сеть, 1 проход"] = lambda t, c: forward_net(t[None].float())[0].detach()
    args.out.mkdir(parents=True, exist_ok=True)
    # Прогрев: первый вызов на ускорителе включает компиляцию ядер и завышает время в 10 раз.
    warm = torch.rand(4, 128, 128, device=dev)
    gerchberg_saxton(warm, chirps, 2)
    fit(PhaseField(random_phase((128, 128), 0)[None]), warm, chirps, EXACT, Schedule(5, 0.3))
    if model is not None:
        synthesize(model, warm, chirps, "fast", 2, 1, forward_net=forward_net if args.ff_init else None)
    synchronize(dev)
    sink = (args.out / f"{args.tag}.jsonl").open("a")
    for folder in args.targets:
        names, images = load_targets(folder)
        targets = torch.from_numpy(images).float().to(dev) / 255
        for method, run in methods.items():
            torch.manual_seed(0)
            start = time.perf_counter()
            phase = run(targets, chirps)
            synchronize(dev)
            seconds = time.perf_counter() - start
            code, recon, metrics = evaluate(phase.reshape(128, 128).cpu().double(), images)
            row = {"set": folder.name, "method": method, "cc": metrics["cc_mean"], "ssim": metrics["ssim_mean"],
                   "cc_min": min(metrics["cc"]), "ssim_min": min(metrics["ssim"]),
                   "brightness": brightness(recon, images), "seconds": seconds}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if args.save_images:
                slug = method.replace(" ", "_").replace("×", "x").replace("+", "_")
                save(args.out / args.tag / folder.name / slug, names, code, recon, metrics,
                     {"method": method, "seconds": seconds})
            print(f"| {row['set']} | {method} | {row['cc']:.3f} | {row['ssim']:.3f} | {row['cc'] + row['ssim']:.3f} | "
                  f"{row['cc_min']:.3f} | {row['ssim_min']:.3f} | {row['brightness']:.2f} | {seconds:.2f} |", flush=True)
    sink.close()


if __name__ == "__main__":
    main()
