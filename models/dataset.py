from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RenderedSceneDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 128,
        limit: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        rows = []
        with (self.root / "metadata.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
                    if limit is not None and len(rows) >= limit:
                        break
        if not rows:
            raise RuntimeError(f"No samples found for split={split!r} in {self.root}")

        self.rows = rows
        self.state_dim = len(self.rows[0]["state"])
        for row in self.rows:
            if len(row["state"]) != self.state_dim:
                raise RuntimeError(
                    f"Mixed state dimensions in {self.root}: expected {self.state_dim}, "
                    f"got {len(row['state'])} for id={row.get('id')}"
                )

        first_images = self.rows[0].get("images", [self.rows[0]["image"]])
        self.views_per_sample = len(first_images)
        if self.views_per_sample < 1:
            raise RuntimeError(f"Dataset row contains no conditioning images: {self.root}")

        for row in self.rows:
            view_paths = row.get("images", [row["image"]])
            if len(view_paths) != self.views_per_sample:
                raise RuntimeError(
                    f"Mixed view counts in {self.root}: expected {self.views_per_sample}, "
                    f"got {len(view_paths)} for id={row.get('id')}"
                )

        # Each view is composited to RGB and views are concatenated channel-wise.
        # v1-v3 therefore remain 3-channel; v4 stereo pairs become 6-channel.
        self.image_channels = 3 * self.views_per_sample

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb_tensor(self, relative_path: str) -> torch.Tensor:
        image = Image.open(self.root / relative_path).convert("RGBA")

        # Composite transparent renders onto a fixed gray background while keeping
        # every individual conditioning view 3-channel RGB.
        bg = Image.new("RGBA", image.size, (32, 32, 32, 255))
        image = Image.alpha_composite(bg, image).convert("RGB")
        return self.transform(image)

    def __getitem__(self, index: int):
        row = self.rows[index]
        view_paths = row.get("images", [row["image"]])
        views = [self._load_rgb_tensor(path) for path in view_paths]
        image_t = torch.cat(views, dim=0)

        state_t = torch.tensor(row["state"], dtype=torch.float32)

        # Keep the batched sample schema uniform. The full metadata row contains
        # shape-dependent / multi-object dicts that default_collate cannot stack.
        return {
            "image": image_t,
            "state": state_t,
            "id": int(row["id"]),
        }
