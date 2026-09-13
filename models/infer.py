from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from br_scene_state import decode_state
from models.dataset import RenderedSceneDataset
from models.diffusion import ConditionalStateDenoiser, GaussianDiffusion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--samples-per-image", type=int, default=8)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ConditionalStateDenoiser().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        steps=int(checkpoint.get("diffusion_steps", 100)),
        device=device,
    )
    image_size = int(checkpoint.get("image_size", 128))
    dataset = RenderedSceneDataset(args.data, args.split, image_size=image_size, limit=args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="sampling"):
            image = batch["image"].to(device)
            states = diffusion.sample(model, image, n=args.samples_per_image).cpu().numpy()
            target_id = int(batch["id"].item())
            target_row = next(row for row in dataset.rows if int(row["id"]) == target_id)

            for sample_index, state in enumerate(states):
                prediction = {
                    "target_id": target_id,
                    "sample_index": sample_index,
                    "target_image": target_row["image"],
                    "target_scene": target_row["scene"],
                    "target_state": target_row["state"],
                    "pred_state": state.tolist(),
                    "pred_scene": decode_state(state),
                }
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
