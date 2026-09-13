from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    raise RuntimeError(
        f"No scene decoder registered for state_dim={state_dim}. "
        "Expected v1=9, v2=34, or v3=35."
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", 128))
    dataset = RenderedSceneDataset(args.data, args.split, image_size=image_size, limit=args.limit)
    state_dim = int(checkpoint.get("state_dim", dataset.state_dim))
    if state_dim != dataset.state_dim:
        raise RuntimeError(
            f"Checkpoint state_dim={state_dim} but dataset state_dim={dataset.state_dim}"
        )

    decode_state = _decoder_for_state_dim(state_dim)
    model = ConditionalStateDenoiser(state_dim=state_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        steps=int(checkpoint.get("diffusion_steps", 100)),
        device=device,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_by_id = {int(row["id"]): row for row in dataset.rows}
    with args.out.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="sampling"):
            image = batch["image"].to(device)
            states = diffusion.sample(model, image, n=args.samples_per_image).cpu().numpy()
            target_id = int(batch["id"].item())
            target_row = row_by_id[target_id]

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
