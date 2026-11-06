"""Синтез ДОЭ обучаемым оптимизатором: один вход для командной строки и для сравнения методов.

Режимы:

- `fast` — один старт, фиксированное число шагов сети. Весь путь от четырёх изображений до фазы
  делает сеть.
- `neural` — несколько стартов в одном пакете, лучший выбирается по той же потере, что считает
  проверка. Тоже целиком нейросетевой: стартовую фазу и каждый шаг даёт сеть.
- `quality` — к ответу сети добавляются шаги обычного градиентного спуска по фазе. Это уже не
  нейросетевой метод, и в отчёте он идёт отдельной строкой.
"""

import time
from dataclasses import dataclass

import torch

from .models.learned import LearnedOptimizer
from .models.phase import PhaseField
from .objective import Objective
from .optimize import Schedule, device, fit, synchronize

FEATURE_OBJECTIVE = Objective(quantize=False)


@dataclass
class Synthesis:
    phase: torch.Tensor
    loss: float
    seconds: float
    steps: int
    refine: int


@torch.no_grad()
def neural_phase(model: LearnedOptimizer, targets: torch.Tensor, chirps: torch.Tensor, steps: int | None = None,
                 seeds: int = 1, select: Objective = Objective(), seed: int = 0,
                 use_init: bool = True, forward_net=None) -> tuple[torch.Tensor, float]:
    """Фаза ДОЭ от сети. targets (P, N, N) в 0..1. Возвращает лучший из seeds стартов и его потерю.

    forward_net — прямая сеть (holo/models/feedforward.py): если она задана, начальную фазу даёт
    она, а не начальная сеть оптимизатора. Её ответ детерминирован, поэтому старт всегда один."""
    dev = chirps.device
    if forward_net is not None:
        seeds = 1
    batch = targets.float().to(dev)[None].expand(seeds, -1, -1, -1).contiguous()
    phase = forward_net(batch) if forward_net is not None else model.start(batch, seed=seed, use_init=use_init)
    history = model(phase, batch, chirps, FEATURE_OBJECTIVE, steps)
    phase = history[-1] if history else phase
    loss = select(phase, chirps, batch)
    index = int(loss.argmin())
    return phase[index], float(loss[index])


def synthesize(model: LearnedOptimizer, targets: torch.Tensor, chirps: torch.Tensor, mode: str = "neural",
               steps: int | None = None, seeds: int | None = None, refine: int | None = None,
               refine_lr: float = 0.2, select: Objective = Objective(), seed: int = 0,
               use_init: bool = True, forward_net=None) -> Synthesis:
    """targets (P, N, N) в 0..1 на любом устройстве; chirps уже на нужном устройстве в complex64."""
    presets = {"fast": (1, 0), "neural": (4, 0), "quality": (4, 500)}
    if mode not in presets:
        raise ValueError(f"режим {mode}: ожидается один из {sorted(presets)}")
    default_seeds, default_refine = presets[mode]
    seeds = default_seeds if seeds is None else seeds
    refine = default_refine if refine is None else refine
    start = time.perf_counter()
    phase, loss = neural_phase(model, targets, chirps, steps, seeds, select, seed, use_init, forward_net)
    used = steps if steps is not None else model.steps
    if refine:
        result = fit(PhaseField(phase[None]), targets, chirps, select,
                     Schedule(refine, refine_lr, quantize_from=0.5))
        phase, loss = result.phase, result.loss
    synchronize(device())
    return Synthesis(phase.reshape(128, 128).cpu().double(), loss, time.perf_counter() - start, used, refine)
