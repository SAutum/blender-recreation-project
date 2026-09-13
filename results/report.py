from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_training(run: Path) -> str | None:
    path = run / "history.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Diffusion noise-prediction MSE")
    plt.legend()
    plt.tight_layout()
    out = run / "training_curve.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out.name


def plot_scores(run: Path) -> str | None:
    path = run / "metrics.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    by_target: dict[int, list[dict]] = {}
    for r in rows:
        by_target.setdefault(int(r["target_id"]), []).append(r)
    best = [max(group, key=lambda r: float(r["main_score"])) for group in by_target.values()]
    scores = [float(r["main_score"]) for r in best]

    plt.figure(figsize=(6.4, 4.0))
    plt.hist(scores, bins=25)
    plt.xlabel("Best-of-K reconstruction score")
    plt.ylabel("Targets")
    plt.tight_layout()
    out = run / "score_histogram.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out.name


def make_examples(run: Path, n_each: int = 3) -> str | None:
    path = run / "metrics.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    by_target: dict[int, list[dict]] = {}
    for r in rows:
        by_target.setdefault(int(r["target_id"]), []).append(r)
    best = [max(group, key=lambda r: float(r["main_score"])) for group in by_target.values()]
    best.sort(key=lambda r: float(r["main_score"]), reverse=True)
    chosen = best[:n_each] + best[-n_each:]
    if not chosen:
        return None

    fig, axes = plt.subplots(len(chosen), 2, figsize=(6.4, 2.5 * len(chosen)))
    if len(chosen) == 1:
        axes = np.asarray([axes])
    for i, r in enumerate(chosen):
        target = Image.open(r["target_image"]).convert("RGBA")
        pred = Image.open(r["pred_image"]).convert("RGBA")
        axes[i, 0].imshow(target)
        axes[i, 1].imshow(pred)
        axes[i, 0].set_title(f"Target {r['target_id']}")
        axes[i, 1].set_title(f"Predicted render, score={float(r['main_score']):.3f}")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    plt.tight_layout()
    out = run / "best_worst_examples.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def tex_escape(value: str) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(ch, ch) for ch in value)


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = {}
    if (run / "run_config.json").exists():
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))

    training_plot = plot_training(run)
    score_plot = plot_scores(run)
    examples_plot = make_examples(run)
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    best = summary["best_of_k"]
    config_rows = "\n".join(
        f"{tex_escape(str(k))} & {tex_escape(str(v))} \\\\" for k, v in sorted(config.items())
    ) or "No run configuration found. & -- \\\\"

    figures = []
    if training_plot:
        figures.append(
            rf"\begin{{figure}}[H]\centering\includegraphics[width=0.82\linewidth]{{{training_plot}}}\caption{{Training and validation diffusion loss.}}\end{{figure}}"
        )
    if score_plot:
        figures.append(
            rf"\begin{{figure}}[H]\centering\includegraphics[width=0.82\linewidth]{{{score_plot}}}\caption{{Distribution of best-of-K image-space reconstruction scores.}}\end{{figure}}"
        )
    if examples_plot:
        figures.append(
            rf"\begin{{figure}}[H]\centering\includegraphics[width=0.92\linewidth]{{{examples_plot}}}\caption{{Highest- and lowest-scoring best-of-K examples.}}\end{{figure}}"
        )

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\title{{Blender Recreation Diffusion Experiment}}
\date{{{tex_escape(generated_at)}}}
\begin{{document}}
\maketitle

\section{{Purpose}}
This experiment tests whether a conditional diffusion model can infer a compact Blender scene state from a rendered image. The predicted state is rendered by Blender again, and image-space agreement with the target is the primary evaluation. Synthetic ground-truth parameter error is retained only as a secondary diagnostic because symmetric primitives may admit multiple valid scene states.

\section{{Run configuration}}
\begin{{tabular}}{{ll}}
\toprule
Field & Value \\
\midrule
{config_rows}
\bottomrule
\end{{tabular}}

\section{{Primary results}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
Targets & {summary['targets']} \\
Total sampled states & {summary['samples_total']} \\
Mean samples per target & {summary['samples_per_target_mean']:.2f} \\
Best-of-K main score mean & {best['main_score_mean']:.4f} \\
Best-of-K main score median & {best['main_score_median']:.4f} \\
Best-of-K silhouette IoU mean & {best['mask_iou_mean']:.4f} \\
Best-of-K SSIM mean & {best['ssim_mean']:.4f} \\
\bottomrule
\end{{tabular}}

\section{{Secondary parameter diagnostics}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
Shape accuracy & {best['shape_accuracy']:.4f} \\
Camera XYZ L2 error & {best['camera_l2_mean']:.4f} \\
Geometry parameter MAE & {best['geometry_mae_mean']:.4f} \\
\bottomrule
\end{{tabular}}

\paragraph{{Interpretation.}}
The image-space score is the main reference. A large camera-parameter error is not necessarily a failure when a symmetric object produces an equivalent rendered observation.

{chr(10).join(figures)}

\section{{Artifacts}}
The run directory contains the raw training history, predictions, per-sample metrics, JSON summary, plots, this LaTeX source, and (when \texttt{{pdflatex}} is available) the compiled PDF. Rendered PNGs and checkpoints remain local and are ignored by git.

\end{{document}}
"""

    tex_path = run / "report.tex"
    tex_path.write_text(tex, encoding="utf-8")

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        for _ in range(2):
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                cwd=run,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        print(f"PDF report: {run / 'report.pdf'}")
    else:
        print("pdflatex not found; report.tex and plots were created, but report.pdf was not compiled.")


if __name__ == "__main__":
    main()
