
from pathlib import Path

import numpy as np
from PIL import Image

from aiml.config import RAW_DATA_DIR, SEGMENTATION_DATA_DIR

IMAGE_NAME = "MCUCXR_0041_0.png"
MASK_NAME = "MCUCXR_0041_0.png"

RAW_DATASET = RAW_DATA_DIR / "archive (2)" / "Lung Segmentation"

ORIGINAL_IMAGE = RAW_DATASET / "CXR_png" / IMAGE_NAME
ORIGINAL_MASK = RAW_DATASET / "masks" / MASK_NAME

PROCESSED_IMAGE = (
    SEGMENTATION_DATA_DIR / "val" / "images" / IMAGE_NAME
)
PROCESSED_MASK = (
    SEGMENTATION_DATA_DIR
    / "val"
    / "masks"
    / f"{Path(IMAGE_NAME).stem}_mask.png"
)


def inspect_image(label, path, is_mask=False):
    if not path.exists():
        print(f"\n{label}: FILE NOT FOUND")
        print(path)
        return None

    with Image.open(path) as image:
        array = np.asarray(image.convert("L")).copy()

    print(f"\n{label}")
    print(f"  Path: {path}")
    print(f"  Dimensions (width, height): {Image.open(path).size}")
    print(f"  Array shape: {array.shape}")
    print(f"  Minimum pixel value: {array.min()}")
    print(f"  Maximum pixel value: {array.max()}")

    if is_mask:
        print(f"  Unique pixel values: {np.unique(array)[:20]}")
        print(f"  Foreground fraction (>0): {(array > 0).mean():.4f}")
        print(f"  Foreground fraction (>=128): {(array >= 128).mean():.4f}")

    return array


def main():
    print("=" * 70)
    print("SEGMENTATION IMAGE-MASK DIAGNOSTIC")
    print("=" * 70)

    original_image = inspect_image(
        "ORIGINAL X-RAY",
        ORIGINAL_IMAGE,
    )

    original_mask = inspect_image(
        "ORIGINAL MASK",
        ORIGINAL_MASK,
        is_mask=True,
    )

    processed_image = inspect_image(
        "PROCESSED X-RAY",
        PROCESSED_IMAGE,
    )

    processed_mask = inspect_image(
        "PROCESSED MASK",
        PROCESSED_MASK,
        is_mask=True,
    )

    arrays = [
        original_image,
        original_mask,
        processed_image,
        processed_mask,
    ]

    if any(array is None for array in arrays):
        print("\nDiagnostic incomplete: one or more files are missing.")
        return

    print("\n" + "=" * 70)
    print("DIMENSION AND AREA COMPARISON")
    print("=" * 70)

    print(
        "Original image and mask dimensions match:",
        original_image.shape == original_mask.shape,
    )

    print(
        "Processed image and mask dimensions match:",
        processed_image.shape == processed_mask.shape,
    )

    print(
        "Original and processed image shapes match:",
        original_image.shape == processed_image.shape,
    )

    print(
        "Original and processed mask shapes match:",
        original_mask.shape == processed_mask.shape,
    )

    print(
        "\nOriginal mask foreground fraction:",
        f"{(original_mask > 0).mean():.4f}",
    )

    print(
        "Processed mask foreground fraction:",
        f"{(processed_mask >= 128).mean():.4f}",
    )

    print("\nNo files were modified.")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/10_check_problem_image.py