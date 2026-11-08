"""Сравнение чужих решений на наших наборах по одной и той же метрике.

Каждое чужое решение запускается собственной командой на четырёх PNG и отдаёт фазовый doe.png.
Дальше ДОЭ читается одним нашим кодом и оценивается так, как требует условие: восстановление
по амплитуде |K| (и отдельно по интенсивности |K|²), нормировка по максимуму, 8 бит, Пирсон и ИСС.

    PYTHONPATH=. python scripts/rival_bench.py --rival meefee --sets photos kodak_1
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from holo.io import load_targets
from holo.metrics import report
from holo.optics import Setup, png_to_phase, reconstruct, to_uint8

RIV = Path("/tmp/rivals")
PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "photos": ROOT / "data/targets/photos", "extra": ROOT / "data/targets/extra",
    "kodak_1": ROOT / "data/heldout/kodak_1", "kodak_2": ROOT / "data/heldout/kodak_2",
    "kodak_3": ROOT / "data/heldout/kodak_3", "diverse": ROOT / "data/heldout/diverse",
    "mixed_1": ROOT / "data/heldout/mixed_1", "mixed_2": ROOT / "data/heldout/mixed_2",
    "glyphs_1": ROOT / "data/heldout/glyphs_1", "similar_photos": ROOT / "data/heldout/similar_photos",
}


def files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def run(cmd, cwd, env_extra=None, timeout=1800):
    import os
    env = {**os.environ, "PYTHONPATH": str(cwd), "PYTHONIOENCODING": "utf-8", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", **(env_extra or {})}
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return time.perf_counter() - t, r


def meefee(folder, out):
    d = RIV / "meefee-teem-doe-synthesis"
    work = Path(tempfile.mkdtemp())
    for i, f in enumerate(files(folder), 1):
        shutil.copy(f, work / f"{i}.png")
    sec, r = run([PY, "infer.py", "--input", str(work), "--output", str(out)], d)
    found = list(out.rglob("doe.png"))
    return sec, r, found[0] if found else None


def donn(folder, out, mode="best"):
    d = RIV / "doe-nn-synthesis"
    work = Path(tempfile.mkdtemp())
    for i, f in enumerate(files(folder), 1):
        shutil.copy(f, work / f"{i}.png")
    sec, r = run([PY, "run.py", "--input", str(work), "--output", str(out), "--mode", mode, "--device", "cpu"], d)
    found = list(out.rglob("doe.png"))
    return sec, r, found[0] if found else None


def donn_fast(folder, out):
    return donn(folder, out, "fast")


def donn_balanced(folder, out):
    return donn(folder, out, "balanced")


def donn_max(folder, out):
    return donn(folder, out, "max")


def fourplane(folder, out, mode="quality"):
    d = RIV / "four-plane-doe-synthesis"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "doe.png"
    sec, r = run([PY, "infer.py", *map(str, files(folder)), "--output", str(target), "--mode", mode], d)
    return sec, r, target if target.exists() else None


def fourplane_fast(folder, out):
    return fourplane(folder, out, "fast")


def mvp(folder, out):
    d = RIV / "MVP-AI-photon-Nuclear-IT-Hack"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "doe.png"
    sec, r = run([PY, "-m", "submission.predict", "--images", *map(str, files(folder)), "--output", str(target)], d)
    return sec, r, target if target.exists() else None


def pinn(folder, out, mode="lens"):
    d = RIV / "AI-Photonics"
    out.mkdir(parents=True, exist_ok=True)
    sec, r = run([PY, "pipeline.py", "--mode", mode, "--images", *map(str, files(folder)), "--out", str(out)], d)
    found = list(out.rglob("*.png"))
    doe = [p for p in found if "doe" in p.name.lower() or "mask" in p.name.lower() or "phase" in p.name.lower()]
    return sec, r, doe[0] if doe else None


def pinn_nolens(folder, out):
    return pinn(folder, out, "no_lens")


def ours(folder, out, mode="neural", extra=()):
    sec, r = run([PY, "scripts/synthesize.py", str(folder), "--out", str(out), "--mode", mode, *extra], ROOT)
    found = list(out.rglob("doe.png"))
    return sec, r, found[0] if found else None


def ours_quality(folder, out):
    return ours(folder, out, "quality")


def doesynth(folder, out):
    d = RIV / "DOESynth"
    out.mkdir(parents=True, exist_ok=True)
    sec, r = run([PY, "-m", "doesynth", "synth", *map(str, files(folder)), "--out", str(out)], d)
    found = list(out.rglob("doe.png"))
    return sec, r, found[0] if found else None


def ours_robust(folder, out):
    return ours(folder, out, "quality", ("--modes", "amplitude", "intensity"))


def ours_w15(folder, out):
    return ours(folder, out, "quality", ("--intensity-weight", "0.15"))


def ours_w30(folder, out):
    return ours(folder, out, "quality", ("--intensity-weight", "0.3"))


def ours_w50(folder, out):
    return ours(folder, out, "quality", ("--intensity-weight", "0.5"))


def ours_w50_r250(folder, out):
    return ours(folder, out, "quality", ("--intensity-weight", "0.5", "--refine", "250"))


def ours_w50_r1000(folder, out):
    return ours(folder, out, "quality", ("--intensity-weight", "0.5", "--refine", "1000"))


def ours_w50_r2000(folder, out):
    return ours(folder, out, "quality", ("--intensity-weight", "0.5", "--refine", "2000"))


RIVALS = {"ours_w50_r2000": ours_w50_r2000, "ours_w50_r250": ours_w50_r250, "ours_w50_r1000": ours_w50_r1000, "ours_w50": ours_w50, "ours_w15": ours_w15, "ours_w30": ours_w30, "ours_robust": ours_robust, "doesynth": doesynth, "ours_neural": ours, "ours_quality": ours_quality, "meefee": meefee, "donn_best": donn, "donn_fast": donn_fast, "donn_balanced": donn_balanced, "donn_max": donn_max, "fourplane": fourplane, "fourplane_fast": fourplane_fast,
          "mvp": mvp, "pinn": pinn, "pinn_nolens": pinn_nolens}


def score(doe_path: Path, targets: np.ndarray) -> dict:
    code = torch.from_numpy(np.asarray(Image.open(doe_path).convert("L"), dtype=np.uint8))
    phase = png_to_phase(code)
    out = {}
    for mode in ("amplitude", "intensity"):
        recon = to_uint8(reconstruct(phase, Setup().chirps(), mode)).numpy()
        r = report(targets, recon)
        out[mode] = {"cc": r["cc_mean"], "ssim": r["ssim_mean"]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rival", required=True, choices=sorted(RIVALS))
    p.add_argument("--sets", nargs="*", default=list(SETS))
    p.add_argument("--out", type=Path, default=Path("/tmp/rival_results"))
    a = p.parse_args()
    results = {}
    for name in a.sets:
        folder = SETS[name]
        names, targets = load_targets(folder)
        outdir = Path(tempfile.mkdtemp())
        sec, r, doe = RIVALS[a.rival](folder, outdir)
        if doe is None:
            print(f"{a.rival} {name}: нет doe.png\n{r.stderr[-600:] or r.stdout[-600:]}")
            results[name] = {"error": (r.stderr or r.stdout)[-300:]}
            continue
        s = score(doe, targets)
        results[name] = {**s, "wall_seconds": round(sec, 2)}
        print(f"{a.rival:14s} {name:15s} ампл. КК {s['amplitude']['cc']:.3f} ИСС {s['amplitude']['ssim']:.3f} | "
              f"интенс. КК {s['intensity']['cc']:.3f} ИСС {s['intensity']['ssim']:.3f} | {sec:.1f} с", flush=True)
    ok = [v for v in results.values() if "amplitude" in v]
    if ok:
        mean = lambda m, k: float(np.mean([v[m][k] for v in ok]))  # noqa: E731
        results["_mean"] = {"amplitude": {"cc": mean("amplitude", "cc"), "ssim": mean("amplitude", "ssim")},
                            "intensity": {"cc": mean("intensity", "cc"), "ssim": mean("intensity", "ssim")},
                            "wall_seconds": float(np.mean([v["wall_seconds"] for v in ok])), "sets": len(ok)}
        print(f"{a.rival:14s} СРЕДНЕЕ         ампл. КК {results['_mean']['amplitude']['cc']:.3f} ИСС {results['_mean']['amplitude']['ssim']:.3f} | "
              f"интенс. КК {results['_mean']['intensity']['cc']:.3f} ИСС {results['_mean']['intensity']['ssim']:.3f}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.rival}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
