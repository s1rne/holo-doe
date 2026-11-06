"""Обучение обучаемого оптимизатора фазы ДОЭ.

Потеря считается ровно так, как проверяет жюри: амплитуда |K|, нормировка по максимуму,
округление до 8 бит, КК и SSIM (`holo.objective.Objective`), фаза тоже квантуется в 8 бит.
Развёрнутые шаги обучаются усечённым обратным проходом: каждый отрезок из `--segment` шагов
получает свой backward, между отрезками фаза отцепляется от графа. Поэтому память не зависит
от полного числа шагов и можно обучать 32–64 шага на 12 ГБ.

Данные: банки data/cache (scripts/make_dataset.py) плюс процедурные сцены, см. holo/dataset.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from holo.dataset import QuadSampler
from holo.evaluate import evaluate
from holo.io import load_targets
from holo.models.learned import LearnedOptimizer, save_model
from holo.objective import Objective
from holo.optics import Setup
from holo.optimize import device

ROOT = Path(__file__).resolve().parent.parent


def eval_sets(folders: list[Path]) -> list[tuple[str, np.ndarray, torch.Tensor]]:
    out = []
    for folder in folders:
        _, images = load_targets(folder)
        out.append((folder.name, images, torch.from_numpy(images).float() / 255))
    return out


@torch.no_grad()
def run_eval(model: LearnedOptimizer, sets, chirps, feature_objective, steps: int) -> dict:
    rows = {}
    for name, images, targets in sets:
        phase = model.start(targets[None].to(chirps.device), seed=0)
        phase = model(phase, targets[None].to(chirps.device), chirps, feature_objective, steps)[-1]
        metrics = evaluate(phase[0].double().cpu(), images)[2]
        rows[name] = (round(metrics["cc_mean"], 4), round(metrics["ssim_mean"], 4))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "neural")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--minutes", type=float, default=None, help="бюджет времени вместо числа итераций")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--segment", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--init-width", type=int, default=24)
    parser.add_argument("--init-depth", type=int, default=3)
    parser.add_argument("--no-init-net", action="store_true", help="стартовать со случайной фазы без сети")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--softmin", type=float, default=0.0)
    parser.add_argument("--worst-weight", type=float, default=1.0)
    parser.add_argument("--mean-weight", type=float, default=0.0)
    parser.add_argument("--speckle-weight", type=float, default=0.0)
    parser.add_argument("--no-quantize", action="store_true", help="потеря без 8-битного квантования")
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()

    dev = device()
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    chirps = Setup().chirps().to(torch.complex64).to(dev)
    sampler = QuadSampler(np.load(args.cache / "natural.npy"), np.load(args.cache / "glyphs.npy"), dev)
    model = LearnedOptimizer(args.steps, args.width, not args.no_init_net, args.init_width, args.init_depth).to(dev)
    if args.init_from:
        state = torch.load(args.init_from, map_location=dev, weights_only=False)["state"]
        print(model.load_state_dict(state, strict=False), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_iters = args.iterations
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_iters, eta_min=args.lr * 0.05)

    feature_objective = Objective(quantize=False)
    train_objective = Objective(quantize=not args.no_quantize, softmin=args.softmin,
                                worst_weight=args.worst_weight, mean_weight=args.mean_weight,
                                speckle_weight=args.speckle_weight)
    weights = torch.linspace(0.1, 1.0, args.steps, device=dev)
    weights[-1] = weights[-1] * 3
    sets = eval_sets([ROOT / "data" / "targets" / "photos", ROOT / "data" / "targets" / "glyphs",
                      ROOT / "data" / "heldout" / "kodak_1", ROOT / "data" / "heldout" / "glyphs_1"])
    (args.out / "config.json").write_text(json.dumps({**vars(args), "out": str(args.out),
                                                      "init_from": str(args.init_from),
                                                      "cache": str(args.cache)}, indent=2))
    log = (args.out / "train.log").open("a")
    start = time.perf_counter()
    for it in range(total_iters):
        targets = sampler.sample(args.batch)
        phase = model.start(targets, seed=it * args.batch)
        first = train_objective(phase, chirps, targets).mean()
        (0.3 * first).backward(retain_graph=True)
        tail = 0.0
        for begin in range(0, args.steps, args.segment):
            if begin:
                phase = phase.detach()
            losses = []
            for k in range(begin, min(begin + args.segment, args.steps)):
                phase = model.step(phase, targets, chirps, feature_objective, k)
                losses.append(train_objective(phase, chirps, targets).mean())
            segment_loss = (weights[begin:begin + len(losses)] * torch.stack(losses)).sum() / weights.sum()
            segment_loss.backward()
            tail = losses[-1].item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if norm.isfinite():
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        elapsed = time.perf_counter() - start
        if it % 25 == 0:
            line = f"{it:6d} start {first.item():.4f} final {tail:.4f} lr {sched.get_last_lr()[0]:.2e} {elapsed / 60:6.1f} мин"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        if args.eval_every and it % args.eval_every == 0 and it:
            rows = run_eval(model, sets, chirps, feature_objective, args.steps)
            line = f"{it:6d} eval " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in rows.items())
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            save_model(model, args.out / "model.pt")
        if args.minutes and elapsed > args.minutes * 60:
            break
    save_model(model, args.out / "model.pt")
    rows = run_eval(model, sets, chirps, feature_objective, args.steps)
    print("final eval " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in rows.items()), flush=True)
    log.write("final eval " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in rows.items()) + "\n")


if __name__ == "__main__":
    main()
