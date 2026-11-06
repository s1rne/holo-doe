# holo-doe

Нейросетевой синтез фазового дифракционного оптического элемента (ДОЭ) 128×128, из которого прямой расчёт дифракции Френеля восстанавливает 4 изображения на расстояниях 24,1; 25,3; 26,5; 27,7 мм. Длина волны 532 нм, пиксель ПВМС 10 мкм. Условие задачи: [docs/task.pdf](docs/task.pdf).

## Запуск

```bash
uv sync
uv run python scripts/make_targets.py
PYTHONPATH=. uv run python scripts/baselines.py data/targets/photos data/targets/glyphs
PYTHONPATH=. uv run python scripts/per_instance.py data/targets/photos data/targets/glyphs data/targets/extra
```

Обучение обучаемого оптимизатора и прогон с ним:

```bash
PYTHONPATH=. uv run python scripts/train_learned_optimizer.py --iterations 1200
PYTHONPATH=. uv run python scripts/per_instance.py data/targets/photos --learned runs/learned_optimizer.pt
```

## Устройство

- `holo/optics.py`: модель Френеля с одним FFT по формуле из условия, кодирование фазы в PNG (0 → −π, 255 → +π).
- `holo/metrics.py`: метрики проверки, коэффициент корреляции Пирсона и SSIM (`skimage`).
- `holo/losses.py`: дифференцируемые версии тех же метрик для обучения.
- `holo/baselines.py`: эталоны для сравнения, многоплоскостной Герчберг–Сэкстон и прямая градиентная оптимизация фазы.
- `holo/evaluate.py`: оценка строго по условию, восстановление считается из сохранённого 8-битного PNG.
- `holo/objective.py`: целевая функция, повторяющая проверку, с квантованием фазы и восстановления через STE.
- `holo/optimize.py`: оптимизация одной сцены любой параметризацией фазы, расписание шага, рестарты в батче.
- `holo/models/`: параметризации фазы (свободное поле, deep image prior, SIREN, Fourier-MLP) и обучаемый оптимизатор.
- `holo/analysis.py`: анализ предела многоплоскостного восстановления, см. [docs/per_instance.md](docs/per_instance.md).

## Проверка модели

```bash
PYTHONPATH=. uv run pytest tests -q
```

Тесты в [tests/test_optics.py](tests/test_optics.py) подтверждают, что реализация совпадает с определением дифракции Френеля: расчёт через БПФ сходится с прямым двойным суммированием по формуле, фаза тонкой линзы собирает свет в центр нужной плоскости, масштаб восстановления отвечает соотношению z = Nδ²/λ (при нём размер пиксела восстановления равен 10 мкм, а z = 24,06 мм — это и есть первая плоскость из условия).

Кодирование фазы: 0 соответствует −π, 255 соответствует +π, шаг 2π/255. Значение ровно +π физически совпадает с −π и кодируется нулём.

