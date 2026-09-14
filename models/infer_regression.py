from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import RenderedSceneDataset
from models.regression import DirectStateRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer Blender states with direct regression baseline")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=500)
    return p.parse_args()


def _decoder_for_state_dim(state_dim: int):
    if state_dim == 9:
        from br_scene_state import decode_state

        return decode_state
    if state_dim == 34:
        from br_scene_state_v2 import decode_state

        return decode_state
    if state_dim == 35:
        from br_scene_state_v3 import decode_state

        return decode_state
    raise RuntimeError(f"No scene decoder registered for state_dim={state_dim}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("model_type") != "direct_regression":
        raise RuntimeError("Checkpoint is not a direct_regression checkpoint")

    image_size = int(checkpoint.get("image_size", 128))
    view_mode = str(checkpoint.get("view_mode", "mono"))
    encoder_type = str(checkpoint.get("encoder_type", "spatial_pair"))

    dataset = RenderedSceneDataset(
        args.data,
        args.split,
        image_size=image_size,
        limit=args.limit,
        view_mode=view_mode,
    )
    state_dim = int(checkpoint.get("state_dim", dataset.state_dim))
    image_channels = int(checkpoint.get("image_channels", dataset.image_channels))

    if state_dim != dataset.state_dim:
        raise RuntimeError(
            f"Checkpoint state_dim={state_dim} but dataset state_dim={dataset.state_dim}"
        )
    if image_channels != dataset.image_channels:
        raise RuntimeError(
            f"Checkpoint image_channels={image_channels} but dataset provides {dataset.image_channels}"
        )

    model = DirectStateRegressor(
        state_dim=state_dim,
        image_channels=image_channels,
        encoder_type=encoder_type,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    decode_state = _decoder_for_state_dim(state_dim)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    row_by_id = {int(row["id"]): row for row in dataset.rows}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(loader, desc="regression"):
                image = batch["image"].to(device)
                state = model(image)[0].cpu().numpy()
                target_id = int(batch["id"].item())
                target_row = row_by_id[target_id]

                source_images = target_row.get("images", [target_row["image"]])
                if view_mode == "mono":
                    conditioning_images = [source_images[0], source_images[0]]
                elif view_mode == "stereo":
                    conditioning_images = [source_images[0], source_images[1]]
                else:
                    conditioning_images = list(source_images)

                prediction = {
                    "target_id": target_id,
                    "sample_index": 0,
                    "target_image": target_row["image"],
                    "conditioning_images": conditioning_images,
                    "conditioning_mode": view_mode,
                    "target_scene": target_row["scene"],
                    "target_state": target_row["state"],
                    "pred_state": state.tolist(),
                    "pred_scene": decode_state(state),
                    "model_type": "direct_regression",
                }
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"Regression predictions written to {args.out}")


if __name__ == "__main__":
    main()
