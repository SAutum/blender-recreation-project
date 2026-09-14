from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_history_csv(history_path: Path, output_path: Path) -> Path:
    """Plot epoch-wise train/validation MSE from a training history CSV."""
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    output_path = Path(output_path)

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with history_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not epochs:
        raise RuntimeError(f"No training history found in {history_path}")

    best_index = min(range(len(val_losses)), key=val_losses.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", markersize=3, linewidth=1.5, label="Train MSE")
    ax.plot(epochs, val_losses, marker="o", markersize=3, linewidth=1.5, label="Validation MSE")
    ax.scatter(
        [epochs[best_index]],
        [val_losses[best_index]],
        marker="*",
        s=110,
        zorder=5,
        label=f"Best val: epoch {epochs[best_index]} ({val_losses[best_index]:.4f})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise prediction MSE")
    ax.set_title("Training / Validation MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    history_path = args.run / "history.csv"
    output_path = args.out or (args.run / "training_curve.png")
    saved = plot_history_csv(history_path, output_path)
    print(f"Training curve written to {saved}")


if __name__ == "__main__":
    main()
