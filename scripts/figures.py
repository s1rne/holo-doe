"""Иллюстрации к отчёту: мозаика восстановлений и кривая «качество — время».

Мозаика собирается из картинок, которые пишет `scripts/bench.py --save-images`: верхняя строка —
исходные изображения, дальше по строке на метод. Подписи строк содержат средние КК и SSIM, чтобы
рядом с картинкой стояло число.

    PYTHONPATH=. uv run python scripts/figures.py montage --bench outputs/bench/main --set glyphs \
        --methods "GD_1000:GD 1000" "GS_50___GD_200:GS 50 + GD 200" --out docs/images/glyphs.png
    PYTHONPATH=. uv run python scripts/figures.py curve --rows outputs/bench/main.jsonl --out docs/images/curve.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from holo.io import load_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_run(folder: Path) -> tuple[np.ndarray, dict]:
    from PIL import Image
    files = sorted(p for p in folder.iterdir() if p.name.startswith("recon_"))
    images = np.stack([np.asarray(Image.open(p).convert("L")) for p in files])
    metrics = json.loads((folder / "metrics.json").read_text())
    return images, metrics


def montage(args) -> None:
    _, targets = load_targets(args.set_folder)
    rows = [("цель", targets, None)]
    for item in args.methods:
        slug, _, label = item.partition(":")
        images, metrics = read_run(args.bench / args.set_folder.name / slug)
        rows.append((label or slug, images, metrics))
    fig, axes = plt.subplots(len(rows), 4, figsize=(4 * 1.7, len(rows) * 1.8))
    axes = np.atleast_2d(axes)
    for r, (label, images, metrics) in enumerate(rows):
        for c in range(4):
            axes[r, c].imshow(images[c], cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if metrics:
                axes[r, c].set_title(f"{metrics['cc'][c]:.2f} / {metrics['ssim'][c]:.2f}", fontsize=7)
            elif r == 0:
                axes[r, c].set_title(f"{(24.1 + 1.2 * c):.1f} мм", fontsize=8)
        text = label if metrics is None else f"{label}\n{metrics['cc_mean']:.3f} / {metrics['ssim_mean']:.3f}"
        axes[r, 0].set_ylabel(text, fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle(args.title or args.set_folder.name, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"записано {args.out}")


FAMILIES = (("GS ", "GS (параллельный)"), ("ping-pong", "ping-pong (последовательный)"),
            ("GS 50 + GD", "GS 50 + GD"), ("GD ", "GD (Adam по фазе)"),
            ("сеть + GD", "сеть + GD"), ("сеть", "обучаемый оптимизатор"))


def family(method: str) -> str | None:
    for prefix, label in FAMILIES:
        if method.startswith(prefix):
            return label
    return None


def curve(args) -> None:
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    if args.sets:
        rows = [r for r in rows if r["set"] in args.sets]
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        name = family(row["method"])
        if name:
            grouped[name][row["method"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, methods in grouped.items():
        points = sorted((np.mean([r["seconds"] for r in v]),
                         np.mean([r["cc"] for r in v]),
                         np.mean([r["ssim"] for r in v])) for v in methods.values())
        t, cc, ss = zip(*points)
        for ax, y in ((axes[0], cc), (axes[1], ss)):
            ax.plot(t, y, "o-", label=name, markersize=4)
    for ax, title in ((axes[0], "КК"), (axes[1], "SSIM")):
        ax.set_xscale("log")
        ax.set_xlabel("время синтеза одной сцены, с")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(args.title or "Качество от времени, оценка по амплитуде |K|", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"записано {args.out}")


GROUPS = {"photos": ("photos",), "glyphs": ("glyphs",), "extra": ("extra",),
          "heldout_kodak": tuple(f"kodak_{i}" for i in range(1, 7)),
          "heldout_glyphs": tuple(f"glyphs_{i}" for i in range(1, 5)),
          "heldout_mixed": ("mixed_1", "mixed_2"),
          "похожие": ("similar_glyphs", "similar_photos"), "разные": ("diverse",)}


def table(args) -> None:
    """Сводная таблица в markdown: строка на метод, колонка на группу наборов."""
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    groups = {k: v for k, v in GROUPS.items() if not args.groups or k in args.groups}
    order, cells = [], defaultdict(dict)
    for row in rows:
        if row["method"] not in order:
            order.append(row["method"])
        for group, members in groups.items():
            if row["set"] in members:
                cells[row["method"]].setdefault(group, []).append(row)
    columns = [g for g in groups if any(g in cells[m] for m in order)]
    value = {"both": lambda r: f"{np.mean([x['cc'] for x in r]):.3f} / {np.mean([x['ssim'] for x in r]):.3f}",
             "min": lambda r: f"{np.mean([x['cc_min'] for x in r]):.3f} / {np.mean([x['ssim_min'] for x in r]):.3f}",
             "brightness": lambda r: f"{np.mean([x['brightness'] for x in r]):.2f}"}[args.value]
    print("| метод | время, с | " + " | ".join(columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for method in order:
        seconds = np.mean([r["seconds"] for group in cells[method].values() for r in group])
        line = [value(cells[method][c]) if c in cells[method] else "" for c in columns]
        print(f"| {method} | {seconds:.2f} | " + " | ".join(line) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("montage")
    m.add_argument("--bench", type=Path, required=True, help="папка outputs/bench/<tag>")
    m.add_argument("--set", dest="set_folder", type=Path, required=True, help="папка с целями")
    m.add_argument("--methods", nargs="+", required=True, help="слаг[:подпись]")
    m.add_argument("--title", default=None)
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=montage)
    c = sub.add_parser("curve")
    c.add_argument("--rows", type=Path, required=True, help="outputs/bench/<tag>.jsonl")
    c.add_argument("--sets", nargs="*", default=None)
    c.add_argument("--title", default=None)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=curve)
    t = sub.add_parser("table")
    t.add_argument("--rows", type=Path, required=True, help="outputs/bench/<tag>.jsonl")
    t.add_argument("--groups", nargs="*", default=None)
    t.add_argument("--value", choices=("both", "min", "brightness"), default="both")
    t.set_defaults(func=table)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
