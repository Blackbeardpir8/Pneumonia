from pathlib import Path

import numpy as np
import tensorflow as tf

from aiml.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEGMENTATION_DATA_DIR,
)

ROOT = Path(SEGMENTATION_DATA_DIR)

def load_grayscale(path):
    data = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        data,
        channels=1,
        expand_animations=False,
    )
    return image.numpy().squeeze()


def main():
    for split in ["train", "val"]:
        image_dir = ROOT / split / "images"
        mask_dir = ROOT / split / "masks"

        mask_by_name = {
            p.stem.removesuffix("_mask") + p.suffix: p
            for p in mask_dir.glob("*.png")
        }

        images = sorted(image_dir.glob("*.png"))

        print(f"\n{'=' * 60}")
        print(f"SPLIT: {split}")
        print(f"Image count: {len(images)}")
        print(f"Mask count: {len(mask_by_name)}")

        foreground_fractions = []
        image_ranges = []
        mask_values_seen = set()

        for image_path in images:
            mask_path = mask_by_name.get(image_path.name)

            if mask_path is None:
                raise ValueError(
                    f"Missing mask for {image_path.name}"
                )

            image = load_grayscale(image_path)
            mask = load_grayscale(mask_path)

            if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(
                    f"Unexpected image shape: "
                    f"{image_path.name}: {image.shape}"
                )

            if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(
                    f"Unexpected mask shape: "
                    f"{mask_path.name}: {mask.shape}"
                )

            if not np.isfinite(image).all():
                raise ValueError(
                    f"Non-finite image values: {image_path.name}"
                )

            if not np.isfinite(mask).all():
                raise ValueError(
                    f"Non-finite mask values: {mask_path.name}"
                )

            foreground = mask >= 128
            fraction = float(foreground.mean())

            foreground_fractions.append(fraction)
            image_ranges.append(
                (float(image.min()), float(image.max()))
            )
            mask_values_seen.update(
                np.unique(mask).tolist()
            )

        print(
            "Mask foreground fraction — "
            f"min: {min(foreground_fractions):.4f}, "
            f"mean: {np.mean(foreground_fractions):.4f}, "
            f"max: {max(foreground_fractions):.4f}"
        )

        print(
            "Image pixel range — "
            f"min across images: "
            f"{min(v[0] for v in image_ranges):.0f}, "
            f"max across images: "
            f"{max(v[1] for v in image_ranges):.0f}"
        )

        unique_mask_values = sorted(mask_values_seen)

        print(
            "Unique mask pixel values "
            "(first 20):",
            unique_mask_values[:20],
        )

        print("Data checks passed for", split)


if __name__ == "__main__":
    main()

# to run this 
# python scripts/05_check_segmentation_data.py