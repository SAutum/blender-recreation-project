from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import RenderedSceneDataset
from models.regression import DirectStateRegressor
from models.training_plot import plot_history_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train direct image -> Blender state regression baseline")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--encoder",
        default="spatial_pair",
        choices=["legacy", "spatial_pair"],
    )
    p.add_argument(
        "--view-mode",
        default="mono",
        choices=["dataset", "mono", "stereo"],
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without a new best val loss. 0 disables.",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            pred = model(image)
            losses.append(float(F.mse_loss(pred, state).item()))
    return float(np.mean(losses))


def save_checkpoint(path: Path, model, args, epoch: int, val_loss: float, state_dim: int, image_channels: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "image_size": args.image_size,
            "state_dim": int(state_dim),
            "image_channels": int(image_channels),
            "encoder_type": args.encoder,
            "view_mode": args.view_mode,
            "model_type": "direct_regression",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = RenderedSceneDataset(args.data, "train", args.image_size, view_mode=args.view_mode)
    val_ds = RenderedSceneDataset(args.data, "val", args.image_size, view_mode=args.view_mode)

    if train_ds.state_dim != val_ds.state_dim:
        raise RuntimeError("Train/val state dimension mismatch")
    if train_ds.image_channels != val_ds.image_channels:
        raise RuntimeError("Train/val image channel mismatch")

    if args.encoder == "spatial_pair" and train_ds.image_channels != 6:
        raise RuntimeError("spatial_pair encoder requires paired input / 6 channels")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DirectStateRegressor(
        state_dim=train_ds.state_dim,
        image_channels=train_ds.image_channels,
        encoder_type=args.encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Model:        direct regression")
    print(f"Encoder:      {args.encoder}")
    print(f"View mode:    {args.view_mode}")
    print(f"Image chans:  {train_ds.image_channels}")
    print(f"Parameters:   {parameter_count:,}")

    run_config = vars(args).copy()
    run_config["data"] = str(run_config["data"])
    run_config["run"] = str(run_config["run"])
    run_config["device"] = str(device)
    run_config["state_dim"] = train_ds.state_dim
    run_config["image_channels"] = train_ds.image_channels
    run_config["parameter_count"] = parameter_count
    run_config["model_type"] = "direct_regression"
    run_config["started_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_config["train_samples"] = len(train_ds)
    run_config["val_samples"] = len(val_ds)
    (args.run / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history_path = args.run / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in bar:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(image)
            loss = F.mse_loss(pred, state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}")

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        save_checkpoint(
            args.run / "last.pt",
            model,
            args,
            epoch,
            val_loss,
            train_ds.state_dim,
            train_ds.image_channels,
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                args.run / "best.pt",
                model,
                args,
                epoch,
                val_loss,
                train_ds.state_dim,
                train_ds.image_channels,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no val improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    print(f"Training complete. Best validation loss: {best_val:.6f}")
    try:
        curve_path = plot_history_csv(history_path, args.run / "training_curve.png")
        print(f"Training curve written to {curve_path}")
    except Exception as exc:
        print(f"Warning: could not create training curve: {exc}")


if __name__ == "__main__":
    main()
