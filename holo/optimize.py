"""Оптимизация одной сцены (per-instance): любая параметризация фазы, потеря Objective,
расписание шага, параллельные рестарты в батче, выбор лучшего рестарта."""

import math
import time
from dataclasses import dataclass, replace

import torch
from torch import nn

from .objective import Objective


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass(frozen=True)
class Schedule:
    steps: int = 2000
    lr: float = 0.05
    warmup: int = 0
    final: float = 0.01
    """Доля начального шага в конце косинусного спада; 1 = постоянный шаг."""
    quantize_from: float = 0.0
    """С какой доли шагов включать STE-квантование фазы и восстановления."""
    seconds: float | None = None
    """Ограничение по времени; шаги расписания тогда отсчитываются по времени."""

    def factor(self, progress: float, step: int) -> float:
        warm = min(1.0, (step + 1) / self.warmup) if self.warmup else 1.0
        return warm * (self.final + (1 - self.final) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


@dataclass
class Result:
    phase: torch.Tensor
    loss: float
    seconds: float
    steps: int
    history: list[float]


def fit(model: nn.Module, targets: torch.Tensor, chirps: torch.Tensor, objective: Objective = Objective(),
        schedule: Schedule = Schedule(), optimizer: str = "adam") -> Result:
    """model() возвращает фазу (B, N, N): B рестартов одной сцены. targets (P, N, N) в 0..1."""
    dev = device()
    model = model.to(dev)
    targets = targets.float().to(dev)
    chirps = chirps.to(torch.complex64).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=schedule.lr) if optimizer == "adam" else torch.optim.SGD(params, lr=schedule.lr, momentum=0.9)
    smooth = replace(objective, quantize=False)
    best_loss, best_phase, history = None, None, []
    start = time.perf_counter()
    step = 0
    while True:
        elapsed = time.perf_counter() - start
        progress = elapsed / schedule.seconds if schedule.seconds else step / schedule.steps
        if progress >= 1:
            break
        for group in opt.param_groups:
            group["lr"] = schedule.lr * schedule.factor(progress, step)
        quantized = objective.quantize and progress >= schedule.quantize_from
        opt.zero_grad(set_to_none=True)
        phase = model()
        loss = (objective if quantized else smooth)(phase, chirps, targets)
        loss.sum().backward()
        opt.step()
        if quantized and (step % 10 == 0 or progress > 0.9):
            with torch.no_grad():
                current = loss.detach()
                if best_loss is None:
                    best_loss, best_phase = current.clone(), phase.detach().clone()
                else:
                    better = current < best_loss
                    best_loss = torch.where(better, current, best_loss)
                    best_phase[better] = phase.detach()[better]
        if step % 50 == 0:
            history.append(float(loss.detach().min()))
        step += 1
    with torch.no_grad():
        phase = model().detach()
        final = objective(phase, chirps, targets)
        if best_loss is None:
            best_loss, best_phase = final, phase
        else:
            better = final < best_loss
            best_loss = torch.where(better, final, best_loss)
            best_phase[better] = phase[better]
        index = int(best_loss.argmin())
    if dev.type == "mps":
        torch.mps.synchronize()
    return Result(best_phase[index].cpu().double(), float(best_loss[index]), time.perf_counter() - start, step, history)
