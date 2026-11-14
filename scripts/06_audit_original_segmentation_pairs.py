from pathlib import Path

import numpy as np
from PIL import Image

from aiml.config import RAW_DATA_DIR

DATASET_ROOT = (
    RAW_DATA_DIR
    / "archive (2)"
    / "Lung Segmentation"
)

IMAGE_DIR = DATASET_ROOT / "CXR_png"
MASK_DIR = DATASET_ROOT / "masks"

EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff",
}


def load_grayscale(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def main():
    mask_by_stem = {}

    for path in sorted(MASK_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        stem = path.stem.removesuffix("_mask")

        if stem in mask_by_stem:
            raise ValueError(
                f"Duplicate mask for {stem}: {path}"
            )

        mask_by_stem[stem] = path

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )

    pairs = [
        (image_path, mask_by_stem[image_path.stem])
        for image_path in image_paths
        if image_path.stem in mask_by_stem
    ]

    print("Matched original pairs:", len(pairs))

    dimension_mismatches = []
    foreground_fractions = []
    mask_values = set()

    for image_path, mask_path in pairs:
        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)

        if image.shape != mask.shape:
            dimension_mismatches.append(
                (
                    image_path.name,
                    image.shape,
                    mask.shape,
                )
            )

        # The preparation script treats every nonzero mask pixel
        # as foreground. Measure that same interpretation here.
        foreground_fractions.append(
            float((mask > 0).mean())
        )

        mask_values.update(np.unique(mask).tolist())

    print(
        "Original image-mask dimension mismatches:",
        len(dimension_mismatches),
    )

    for name, image_shape, mask_shape in (
        dimension_mismatches[:20]
    ):
        print(
            f"  {name}: image={image_shape}, "
            f"mask={mask_shape}"
        )

    if foreground_fractions:
        print(
            "Original mask foreground fraction — "
            f"min: {min(foreground_fractions):.4f}, "
            f"mean: {np.mean(foreground_fractions):.4f}, "
            f"max: {max(foreground_fractions):.4f}"
        )

    print(
        "Original mask pixel values (first 30):",
        sorted(mask_values)[:30],
    )

    print("\nAudit completed. No files were modified.")


if __name__ == "__main__":
    main()


# to run this script: python scripts/06_audit_original_segmentation_pairs.py