from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument(
        "--targets-from",
        type=Path,
        required=True,
        help="Diffusion predictions.jsonl; target ids and K are inferred from it.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    diffusion_rows = read_jsonl(args.targets_from)
    if not diffusion_rows:
        raise RuntimeError("No diffusion prediction rows found")

    by_target = defaultdict(list)
    for row in diffusion_rows:
        by_target[int(row["target_id"])].append(row)

    dataset_rows = read_jsonl(args.data / "metadata.jsonl")
    row_by_id = {int(row["id"]): row for row in dataset_rows}
    train_pool = [row for row in dataset_rows if row.get("split") == "train"]
    if not train_pool:
        raise RuntimeError("No train rows available for random-state baseline")

    random_rows: list[dict] = []
    gt_rows: list[dict] = []

    for target_id in sorted(by_target):
        if target_id not in row_by_id:
            raise KeyError(f"Target id {target_id} not found in dataset metadata")

        target = row_by_id[target_id]
        k = len(by_target[target_id])

        # Random-valid-state baseline: sample complete scene states from the same
        # training distribution, without conditioning on the target image.
        for sample_index in range(k):
            donor = rng.choice(train_pool)
            random_rows.append(
                {
                    "target_id": target_id,
                    "sample_index": sample_index,
                    "target_image": target["image"],
                    "target_scene": target["scene"],
                    "target_state": target["state"],
                    "pred_state": donor["state"],
                    "pred_scene": donor["scene"],
                    "baseline": "random_valid_state",
                    "donor_id": int(donor["id"]),
                }
            )

        # GT rerender sanity check: the exact Blender state that generated target.
        gt_rows.append(
            {
                "target_id": target_id,
                "sample_index": 0,
                "target_image": target["image"],
                "target_scene": target["scene"],
                "target_state": target["state"],
                "pred_state": target["state"],
                "pred_scene": target["scene"],
                "baseline": "ground_truth_rerender",
            }
        )

    random_path = args.out / "random_predictions.jsonl"
    gt_path = args.out / "gt_predictions.jsonl"
    write_jsonl(random_path, random_rows)
    write_jsonl(gt_path, gt_rows)

    print(f"Targets: {len(by_target)}")
    print(f"Random baseline rows: {len(random_rows)} -> {random_path}")
    print(f"GT rerender rows: {len(gt_rows)} -> {gt_path}")


if __name__ == "__main__":
    main()
