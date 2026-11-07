"""Итоговая команда условия: из папки с 4 PNG синтезировать ДОЭ и посчитать метрики.

    PYTHONPATH=. uv run python scripts/synthesize.py <папка с 4 PNG> --out <папка>

В выходной папке:

- `doe.png` — фаза ДОЭ, 128×128, 8 бит, один канал; 0 соответствует −π, 255 соответствует +π;
- `recon_<имя>_<z>mm.png` — 4 восстановления, 128×128, 8 бит, один канал; восстановление
  считается из сохранённого `doe.png` прямым расчётом дифракции Френеля с одним БПФ, берётся
  амплитуда |K| и нормируется по максимуму;
- `metrics.json` — коэффициент корреляции Пирсона и индекс структурного сходства по каждой паре
  «исходное изображение — восстановление», их средние, время синтеза и параметры режима.

Изображения сопоставляются плоскостям 24,1; 25,3; 26,5; 27,7 мм в порядке сортировки имён файлов.

Режимы (`--mode`):

- `neural` (по умолчанию) — обучаемый оптимизатор: начальную фазу и все шаги даёт сеть, из
  четырёх параллельных стартов выбирается лучший по той же потере, что считает проверка;
- `fast` — один старт, минимальное время;
- `quality` — к ответу сети добавляются `--refine` шагов обычного градиентного спуска по фазе.
  Этот режим уже не нейросетевой и приводится отдельно.

Ключ `--balance` в режиме `quality` включает ограничение на распределение энергии: он запрещает
решению набирать SSIM за счёт того, что фон после нормировки по максимуму округляется в ноль.
На изображениях с чёрным фоном это заметно меняет картинку и заметно снижает метрики; разбор и
числа в docs/neural_method.md.
"""

import argparse
import time
from pathlib import Path

import torch

from holo.evaluate import evaluate, save
from holo.io import load_targets
from holo.models.learned import load_model
from holo.neural import synthesize
from holo.objective import Objective
from holo.optics import Setup
from holo.optimize import device

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "learned_optimizer.pt"
DEFAULT_FORWARD = ROOT / "models" / "initializer.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", type=Path, help="папка с 4 файлами PNG 128×128, 8 бит, один канал")
    parser.add_argument("--out", type=Path, required=True, help="папка для doe.png, восстановлений и metrics.json")
    parser.add_argument("--mode", choices=("neural", "fast", "quality"), default="neural")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ff-checkpoint", type=Path, default=DEFAULT_FORWARD,
                        help="чекпойнт прямой сети: начальную фазу даёт она, а не начальная сеть оптимизатора")
    parser.add_argument("--steps", type=int, default=None, help="число шагов сети, по умолчанию как при обучении")
    parser.add_argument("--seeds", type=int, default=None, help="число параллельных стартов")
    parser.add_argument("--refine", type=int, default=None, help="шагов градиентного спуска после сети")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance", type=float, default=0.0,
                        help="вес штрафа за недобор средней яркости; действует в режиме quality. "
                             "Значение 3 даёт изображения, которые читаются глазом, ценой метрик")
    parser.add_argument("--balance-floor", type=float, default=0.5,
                        help="порог средней яркости, ниже которого включается штраф --balance")
    args = parser.parse_args()

    names, images = load_targets(args.targets)
    targets = torch.from_numpy(images).double() / 255
    dev = device()
    chirps = Setup().chirps().to(torch.complex64).to(dev)
    model = load_model(args.checkpoint, dev)
    forward_net = None
    if args.ff_checkpoint:
        from holo.models.feedforward import load_model as load_forward
        forward_net = load_forward(args.ff_checkpoint, dev)[0]
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    select = Objective(mean_weight=args.balance, mean_floor=args.balance_floor)
    result = synthesize(model, targets, chirps, args.mode, args.steps, args.seeds, args.refine,
                        select=select, seed=args.seed, forward_net=forward_net)
    seconds = time.perf_counter() - start
    code, recon, metrics = evaluate(result.phase, images)
    extra = {"method": "learned-opt", "mode": args.mode, "steps": result.steps, "refine": result.refine,
             "seconds": seconds, "checkpoint": str(args.checkpoint), "ff_checkpoint": str(args.ff_checkpoint),
             "balance": args.balance, "balance_floor": args.balance_floor, "device": dev.type}
    save(args.out, names, code, recon, metrics, extra)
    for name, cc, ss in zip(names, metrics["cc"], metrics["ssim"]):
        print(f"{name:24} КК {cc:.4f}  ИСС {ss:.4f}")
    print(f"{'среднее':24} КК {metrics['cc_mean']:.4f}  ИСС {metrics['ssim_mean']:.4f}  {seconds:.2f} с")
    print(f"записано в {args.out}")


if __name__ == "__main__":
    main()
