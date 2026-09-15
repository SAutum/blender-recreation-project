from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from matplotlib import cm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dataset import RenderedSceneDataset
from models.regression import DirectStateRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize what a direct-regression model uses with Grad-CAM."
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=8)
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional comma-separated dataset ids. Overrides --limit selection.",
    )
    p.add_argument(
        "--target",
        default="all",
        help=(
            "Grad-CAM target: all, camera, object1, object2, count, or index:N. "
            "For 'all' each state dimension gets equal weight after per-dimension CAM normalization."
        ),
    )
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    return p.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def target_indices(target: str, state_dim: int) -> list[int]:
    target = target.strip().lower()
    if target == "all":
        return list(range(state_dim))
    if target.startswith("index:"):
        index = int(target.split(":", 1)[1])
        if not 0 <= index < state_dim:
            raise ValueError(f"State index {index} is outside [0, {state_dim - 1}]")
        return [index]

    if state_dim != 35:
        raise ValueError(
            f"Named target group {target!r} is only defined for the current 35-D v3 state. "
            "Use --target all or --target index:N instead."
        )

    groups = {
        "count": [0],
        "camera": list(range(1, 9)),
        "object1": list(range(9, 22)),
        "object2": list(range(22, 35)),
    }
    if target not in groups:
        raise ValueError(
            f"Unknown --target {target!r}; expected all, camera, object1, object2, count, or index:N"
        )
    return groups[target]


def target_feature_module(model: DirectStateRegressor):
    encoder = model.image_encoder
    if model.encoder_type == "spatial_pair":
        # Shared branch output: B x 128 x 8 x 8 for 128 px input.
        return encoder.shared
    if model.encoder_type == "attention_pair":
        # Shared CNN output before flattening into Transformer tokens.
        return encoder.shared_cnn
    if model.encoder_type == "legacy":
        # Last spatial activation before AdaptiveAvgPool2d(1).
        return encoder.net[7]
    raise ValueError(f"Grad-CAM does not know encoder_type={model.encoder_type!r}")


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    max_value = cam.max()
    if float(max_value) > 1e-12:
        cam = cam / max_value
    return cam


def gradcam_for_output_group(
    prediction: torch.Tensor,
    activations: list[torch.Tensor],
    indices: list[int],
) -> list[torch.Tensor]:
    """Average magnitude Grad-CAM over selected regression outputs.

    Regression outputs do not have a class-positive direction like classification.
    We therefore use the magnitude of the channel-weighted activation map, then
    normalize each output dimension independently before averaging. This answers
    'where is this group of state predictions sensitive?' rather than implying a
    positive/negative causal direction.
    """
    accumulators = [torch.zeros_like(a[0, 0]) for a in activations]

    for output_index in indices:
        grads = torch.autograd.grad(
            prediction[0, output_index],
            activations,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        for i, (activation, grad) in enumerate(zip(activations, grads)):
            weights = grad.mean(dim=(2, 3), keepdim=True)
            raw_cam = (weights * activation).sum(dim=1)[0].abs()
            accumulators[i] = accumulators[i] + normalize_cam(raw_cam.detach())

    return [normalize_cam(cam / max(len(indices), 1)) for cam in accumulators]


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def resize_cam(cam: torch.Tensor, height: int, width: int) -> np.ndarray:
    cam4 = cam.detach().float().cpu()[None, None]
    resized = F.interpolate(cam4, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return resized.clamp(0.0, 1.0).numpy()


def make_overlay(rgb: np.ndarray, cam_values: np.ndarray, alpha: float) -> np.ndarray:
    cmap = cm.get_cmap("inferno")
    heat_rgb = np.round(cmap(cam_values)[..., :3] * 255.0).astype(np.uint8)
    mixed = (1.0 - alpha) * rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def labeled_panel(images: list[tuple[str, np.ndarray]]) -> Image.Image:
    pil_images = [Image.fromarray(array, mode="RGB") for _, array in images]
    width = max(image.width for image in pil_images)
    height = max(image.height for image in pil_images)
    label_h = 26
    canvas = Image.new("RGB", (width * len(pil_images), height + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i, ((label, _), image) in enumerate(zip(images, pil_images)):
        x = i * width
        canvas.paste(image, (x, label_h))
        draw.text((x + 6, 6), label, fill=(240, 240, 240))
    return canvas


def selected_dataset_indices(dataset: RenderedSceneDataset, ids: str | None, limit: int) -> list[int]:
    if ids:
        wanted = [int(value.strip()) for value in ids.split(",") if value.strip()]
        by_id = {int(row["id"]): index for index, row in enumerate(dataset.rows)}
        missing = [sample_id for sample_id in wanted if sample_id not in by_id]
        if missing:
            raise KeyError(f"Requested ids are not present in split={dataset.split!r}: {missing}")
        return [by_id[sample_id] for sample_id in wanted]
    return list(range(min(int(limit), len(dataset))))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")

    data = resolve_path(args.data)
    run_dir = resolve_path(args.run)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = run_dir / "best.pt"
    else:
        checkpoint_path = resolve_path(checkpoint_path)
    out_dir = resolve_path(args.out) if args.out is not None else (run_dir / "gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_type") != "direct_regression":
        raise RuntimeError("Grad-CAM tool currently expects a direct_regression checkpoint")

    image_size = int(checkpoint.get("image_size", 128))
    state_dim = int(checkpoint["state_dim"])
    image_channels = int(checkpoint["image_channels"])
    encoder_type = str(checkpoint.get("encoder_type", "spatial_pair"))
    view_mode = str(checkpoint.get("view_mode", "mono"))

    dataset = RenderedSceneDataset(
        data,
        split=args.split,
        image_size=image_size,
        view_mode=view_mode,
    )
    model = DirectStateRegressor(
        state_dim=state_dim,
        image_channels=image_channels,
        encoder_type=encoder_type,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    indices = target_indices(args.target, state_dim)
    sample_indices = selected_dataset_indices(dataset, args.ids, args.limit)
    feature_module = target_feature_module(model)

    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Encoder:     {encoder_type}")
    print(f"View mode:   {view_mode}")
    print(f"Target:      {args.target} -> state indices {indices}")
    print(f"Samples:     {len(sample_indices)}")
    print(f"Output:      {out_dir}")

    results: list[dict] = []

    for dataset_index in sample_indices:
        sample = dataset[dataset_index]
        image = sample["image"].unsqueeze(0).to(device)
        target_id = int(sample["id"])

        activations: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError("Grad-CAM target layer did not return a BCHW tensor")
            activations.append(output)

        handle = feature_module.register_forward_hook(forward_hook)
        try:
            model.zero_grad(set_to_none=True)
            prediction = model(image)
            if not activations:
                raise RuntimeError("No Grad-CAM activations were captured")
            cams = gradcam_for_output_group(prediction, activations, indices)
        finally:
            handle.remove()

        view_tensors = [image[0, :3]]
        view_labels = ["left"]
        if image.shape[1] >= 6:
            view_tensors.append(image[0, 3:6])
            view_labels.append("right")

        # Paired encoders invoke the shared feature module twice. Legacy encoder
        # invokes it once. Keep visualization aligned with captured branch count.
        if len(cams) != len(view_tensors):
            if len(cams) == 1:
                view_tensors = view_tensors[:1]
                view_labels = view_labels[:1]
            else:
                raise RuntimeError(
                    f"Captured {len(cams)} feature maps but have {len(view_tensors)} input views"
                )

        panels: list[tuple[str, np.ndarray]] = []
        saved_views: list[dict] = []
        for view_index, (label, view_tensor, cam_tensor) in enumerate(
            zip(view_labels, view_tensors, cams)
        ):
            rgb = tensor_to_rgb(view_tensor)
            cam_values = resize_cam(cam_tensor, rgb.shape[0], rgb.shape[1])
            overlay = make_overlay(rgb, cam_values, args.overlay_alpha)

            panels.append((f"{label} input", rgb))
            panels.append((f"{label} Grad-CAM", overlay))

            heat_uint8 = np.round(cam_values * 255.0).astype(np.uint8)
            heat_path = out_dir / f"target_{target_id:07d}_{args.target}_{label}_heat.png"
            overlay_path = out_dir / f"target_{target_id:07d}_{args.target}_{label}_overlay.png"
            Image.fromarray(heat_uint8, mode="L").save(heat_path)
            Image.fromarray(overlay, mode="RGB").save(overlay_path)
            saved_views.append(
                {
                    "view": label,
                    "heatmap": heat_path.name,
                    "overlay": overlay_path.name,
                }
            )

        panel = labeled_panel(panels)
        panel_path = out_dir / f"target_{target_id:07d}_{args.target}_panel.png"
        panel.save(panel_path)

        results.append(
            {
                "target_id": target_id,
                "dataset_index": dataset_index,
                "target": args.target,
                "state_indices": indices,
                "prediction": prediction[0].detach().cpu().tolist(),
                "panel": panel_path.name,
                "views": saved_views,
            }
        )
        print(f"Saved target {target_id}: {panel_path.name}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "data": str(data),
        "split": args.split,
        "encoder_type": encoder_type,
        "view_mode": view_mode,
        "state_dim": state_dim,
        "target": args.target,
        "state_indices": indices,
        "method": (
            "Magnitude Grad-CAM on the last shared spatial CNN feature map. "
            "For multi-output groups, each output dimension is normalized separately "
            "and the CAMs are averaged with equal weight."
        ),
        "results": results,
    }
    summary_path = out_dir / f"summary_{args.target}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
