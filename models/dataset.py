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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGBA")

        # Composite transparent renders onto a fixed gray background while keeping
        # the actual training image 3-channel RGB.
        bg = Image.new("RGBA", image.size, (32, 32, 32, 255))
        image = Image.alpha_composite(bg, image).convert("RGB")
        image_t = self.transform(image)

        state_t = torch.tensor(row["state"], dtype=torch.float32)

        # Keep the batched sample schema uniform. The full metadata row contains
        # shape-dependent / multi-object dicts that default_collate cannot stack.
        return {
            "image": image_t,
            "state": state_t,
            "id": int(row["id"]),
        }
