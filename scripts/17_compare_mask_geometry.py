
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path("data/processed/segmentation")


def inspect_mask(mask_path):
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) >= 128

    height, width = mask.shape
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError(f"Empty mask: {mask_path}")

    components, count = ndimage.label(mask)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return {
        "source": mask_path.name.split("_")[0],
        "bbox_width": (x_max - x_min + 1) / width,
        "bbox_height": (y_max - y_min + 1) / height,
        "bbox_area": (
            (x_max - x_min + 1) *
            (y_max - y_min + 1) /
            (width * height)
        ),
        "center_x": xs.mean() / width,
        "center_y": ys.mean() / height,
        "components": count,
    }


def main():
    for split in ["train", "val"]:
        mask_dir = ROOT / split / "masks"

        rows = [
            inspect_mask(path)
            for path in sorted(mask_dir.glob("*_mask.png"))
        ]

        df = pd.DataFrame(rows)

        print(f"\n{split.upper()} MASK GEOMETRY")
        print("=" * 65)

        print(
            df.groupby("source")[
                [
                    "bbox_width",
                    "bbox_height",
                    "bbox_area",
                    "center_x",
                    "center_y",
                    "components",
                ]
            ].agg(["mean", "std"]).round(4).to_string()
        )


if __name__ == "__main__":
    main()


# to run script, use the following command:
# python scripts/17_compare_mask_geometry.py