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
    p.add_argument(
        "--shuffle-conditioning",
        action="store_true",
        help=(
            "Condition each target on the next dataset sample instead of its own image(s). "
            "This is a deterministic ablation for testing whether the model actually uses images."
        ),
    )
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
        "Expected v1=9, v2=34, or v3/v4/v5=35."
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", 128))
    view_mode = str(checkpoint.get("view_mode", "dataset"))
    encoder_type = str(checkpoint.get("encoder_type", "legacy"))
    dataset = RenderedSceneDataset(
        args.data,
        args.split,
        image_size=image_size,
        limit=args.limit,
        view_mode=view_mode,
    )
    state_dim = int(checkpoint.get("state_dim", dataset.state_dim))
    image_channels = int(checkpoint.get("image_channels", 3))

    if state_dim != dataset.state_dim:
        raise RuntimeError(
            f"Checkpoint state_dim={state_dim} but dataset state_dim={dataset.state_dim}"
        )
    if image_channels != dataset.image_channels:
        raise RuntimeError(
            f"Checkpoint image_channels={image_channels} but dataset provides "
            f"{dataset.image_channels}. Use a checkpoint trained on the same view mode."
        )
    if args.shuffle_conditioning and len(dataset) < 2:
        raise RuntimeError("Shuffled conditioning needs at least two samples")

    decode_state = _decoder_for_state_dim(state_dim)
    model = ConditionalStateDenoiser(
        state_dim=state_dim,
        image_channels=image_channels,
        encoder_type=encoder_type,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        steps=int(checkpoint.get("diffusion_steps", 100)),
        device=device,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_by_id = {int(row["id"]): row for row in dataset.rows}
    mode = "shuffled conditioning" if args.shuffle_conditioning else "normal conditioning"
    print(f"Inference mode: {mode}")
    print(f"Encoder:        {encoder_type}")
    print(f"View mode:      {view_mode}")

    with args.out.open("w", encoding="utf-8") as f:
        for dataset_index, batch in enumerate(tqdm(loader, desc="sampling")):
            target_id = int(batch["id"].item())
            target_row = row_by_id[target_id]

            if args.shuffle_conditioning:
                conditioning_index = (dataset_index + 1) % len(dataset)
                conditioning_sample = dataset[conditioning_index]
                image = conditioning_sample["image"].unsqueeze(0).to(device)
                conditioning_id = int(conditioning_sample["id"])
                conditioning_row = row_by_id[conditioning_id]
            else:
                image = batch["image"].to(device)
                conditioning_id = target_id
                conditioning_row = target_row

            states = diffusion.sample(model, image, n=args.samples_per_image).cpu().numpy()

            for sample_index, state in enumerate(states):
                prediction = {
                    "target_id": target_id,
                    "sample_index": sample_index,
                    "target_image": target_row["image"],
                    "conditioning_target_id": conditioning_id,
                    "conditioning_images": conditioning_row.get(
                        "images", [conditioning_row["image"]]
                    ),
                    "conditioning_view_mode": view_mode,
                    "conditioning_encoder": encoder_type,
                    "conditioning_mode": (
                        "shuffled" if args.shuffle_conditioning else "normal"
                    ),
                    "target_scene": target_row["scene"],
                    "target_state": target_row["state"],
                    "pred_state": state.tolist(),
                    "pred_scene": decode_state(state),
                }
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
