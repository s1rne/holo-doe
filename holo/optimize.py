"""Оптимизация одной сцены (per-instance): любая параметризация фазы, потеря Objective,
расписание шага, параллельные рестарты в батче, выбор лучшего рестарта."""

import math
import time
from dataclasses import dataclass, replace

import torch
from torch import nn

from .objective import Objective


def synchronize(dev: torch.device) -> None:
    """Дождаться конца работы ускорителя: и CUDA, и MPS ставят операции в очередь, без этого
    измеренное время показывает постановку в очередь, а не счёт."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass(frozen=True)
class Schedule:
    steps: int = 2000
    lr: float = 0.05
    final: float = 0.01
    """Доля начального шага в конце косинусного спада; 1 = постоянный шаг."""
    quantize_from: float = 0.0
    """С какой доли шагов включать STE-квантование фазы и восстановления."""
    seconds: float | None = None
    """Ограничение по времени вместо числа шагов; расписание тогда идёт по времени."""

    def factor(self, progress: float) -> float:
        return self.final + (1 - self.final) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@dataclass
class Result:
    phase: torch.Tensor
    loss: float
    seconds: float
    steps: int


def fit(model: nn.Module, targets: torch.Tensor, chirps: torch.Tensor, objective: Objective = Objective(),
        schedule: Schedule = Schedule()) -> Result:
    """model() возвращает фазу (B, N, N): B рестартов одной сцены. targets (P, N, N) в 0..1."""
    dev = device()
    model = model.to(dev)
    targets = targets.float().to(dev)
    chirps = chirps.to(torch.complex64).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=schedule.lr)
    smooth = replace(objective, quantize=False)
    best_loss, best_phase = None, None
    start = time.perf_counter()
    step = 0
    while True:
        elapsed = time.perf_counter() - start
        progress = elapsed / schedule.seconds if schedule.seconds else step / schedule.steps
        if progress >= 1:
            break
        for group in opt.param_groups:
            group["lr"] = schedule.lr * schedule.factor(progress)
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
    synchronize(dev)
    return Result(best_phase[index].cpu().double(), float(best_loss[index]), time.perf_counter() - start, step)
