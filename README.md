# holo-doe

Нейросетевой синтез фазового дифракционного оптического элемента (ДОЭ) 128×128, из которого прямой расчёт дифракции Френеля восстанавливает 4 изображения на расстояниях 24,1; 25,3; 26,5; 27,7 мм. Длина волны 532 нм, пиксель ПВМС 10 мкм. Условие задачи: [docs/task.pdf](docs/task.pdf).

## Синтез ДОЭ

```bash
uv sync
PYTHONPATH=. uv run python scripts/synthesize.py <папка с 4 PNG> --out <папка>
```

В выходной папке: `doe.png` (128×128, 8 бит, один канал, 0 → −π, 255 → +π), четыре восстановления `recon_<имя>_<z>mm.png` и `metrics.json` с коэффициентом корреляции Пирсона и индексом структурного сходства по каждой паре. Изображения сопоставляются плоскостям в порядке сортировки имён файлов.

Режимы: `--mode neural` (по умолчанию, целиком нейросетевой), `--mode fast` (один старт), `--mode quality` (ответ сети плюс шаги обычного градиентного спуска, уже не нейросетевой). Метод, архитектура и измерения: [docs/neural_method.md](docs/neural_method.md).

## Воспроизведение

```bash
uv run python scripts/make_targets.py
PYTHONPATH=. uv run python scripts/make_heldout.py
PYTHONPATH=. uv run python scripts/make_dataset.py          # банки обучения в data/cache
PYTHONPATH=. uv run python scripts/train_learned_optimizer.py --iterations 5000 --out runs/base
PYTHONPATH=. uv run python scripts/bench.py data/targets/* data/heldout/* --checkpoint runs/base/model.pt
PYTHONPATH=. uv run python scripts/per_instance.py data/targets/photos data/targets/glyphs data/targets/extra
```

## Устройство

- `holo/optics.py`: модель Френеля с одним FFT по формуле из условия, кодирование фазы в PNG (0 → −π, 255 → +π).
- `holo/metrics.py`: метрики проверки, коэффициент корреляции Пирсона и SSIM (`skimage`).
- `holo/losses.py`: дифференцируемые версии тех же метрик плюс мера спекла для обучения.
- `holo/objective.py`: целевая функция, повторяющая проверку, с квантованием фазы и восстановления через STE, весом худшей плоскости и штрафами против вырожденного решения.
- `holo/baselines.py`: классические эталоны — параллельный многоплоскостной Герчберг–Сэкстон, последовательный ping-pong и прямая градиентная оптимизация фазы.
- `holo/evaluate.py`: оценка строго по условию, восстановление считается из сохранённого 8-битного PNG.
- `holo/optimize.py`: оптимизация одной сцены любой параметризацией фазы, расписание шага, рестарты в батче.
- `holo/models/learned.py`: обучаемый оптимизатор — основной метод; `holo/models/feedforward.py`: прямая сеть; остальные файлы в `holo/models/` — параметризации фазы для сравнения (свободное поле, deep image prior, SIREN, Fourier-MLP).
- `holo/neural.py`: синтез обучаемым оптимизатором, общий для командной строки и сравнения методов.
- `holo/dataset.py`: банки обучающих изображений и случайные четвёрки; `holo/pretrain.py`: этап гладкой фазы; `holo/analysis.py`: анализ предела многоплоскостного восстановления.

Документы: [docs/neural_method.md](docs/neural_method.md) — нейросетевой метод и сравнение с классическими; [docs/per_instance.md](docs/per_instance.md) — синтез под одну сцену и потолок качества; [docs/feedforward.md](docs/feedforward.md) — прямая сеть; [docs/research.md](docs/research.md) — обзор литературы.

## Проверка модели

```bash
PYTHONPATH=. uv run pytest tests -q
```

Тесты в [tests/test_optics.py](tests/test_optics.py) подтверждают, что реализация совпадает с определением дифракции Френеля: расчёт через БПФ сходится с прямым двойным суммированием по формуле, фаза тонкой линзы собирает свет в центр нужной плоскости, масштаб восстановления отвечает соотношению z = Nδ²/λ (при нём размер пиксела восстановления равен 10 мкм, а z = 24,06 мм — это и есть первая плоскость из условия). [tests/test_objective.py](tests/test_objective.py) проверяет штрафы против вырожденного решения.

Кодирование фазы: 0 соответствует −π, 255 соответствует +π, шаг 2π/255. Значение ровно +π физически совпадает с −π и кодируется нулём.
