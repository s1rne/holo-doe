"""Обучение обучаемого оптимизатора на случайных сценах (holo.data.TrainScenes)."""

import argparse
import time
from pathlib import Path

import torch

from holo.baselines import random_phase
from holo.data import TrainScenes
from holo.models.learned import LearnedOptimizer
from holo.objective import Objective
from holo.optics import Setup
from holo.optimize import device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("runs/learned_optimizer.pt"))
    args = parser.parse_args()
    dev = device()
    torch.manual_seed(0)
    chirps = Setup().chirps().to(torch.complex64).to(dev)
    scenes = TrainScenes()
    model = LearnedOptimizer(args.steps).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iterations)
    objective = Objective(quantize=False, peak_power=16)
    weights = torch.linspace(0.1, 1, args.steps, device=dev)
    start = time.perf_counter()
    for it in range(args.iterations):
        targets = scenes.batch(args.batch).to(dev)
        phase = torch.stack([random_phase((128, 128), it * args.batch + b) for b in range(args.batch)]).float().to(dev)
        history = model(phase, targets, chirps, objective)
        losses = torch.stack([objective(p, chirps, targets).mean() for p in history])
        loss = (weights * losses).sum() / weights.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 50 == 0:
            print(f"{it:5d} final {losses[-1].item():.4f} first {losses[0].item():.4f} {time.perf_counter() - start:6.0f} с", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "steps": args.steps}, args.out)


if __name__ == "__main__":
    main()
