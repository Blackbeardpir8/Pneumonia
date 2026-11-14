"""
Prepare the lung segmentation dataset for Lightweight U-Net training.

Features:
1. Matches X-ray images with their masks by filename.
2. Splits matched pairs into train, val, and test sets.
3. Resizes images and masks to the configured image dimensions.
4. Saves processed images and binary masks separately.
5. Creates a CSV manifest containing the split assignments.

IMPORTANT:
- Original files under data/raw/ are never modified.
- Only matched image-mask pairs are processed.
- A fixed random seed makes the split reproducible.
"""

from pathlib import Path
import csv
import random
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    RAW_DATA_DIR,
    SEGMENTATION_DATA_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = (
    RAW_DATA_DIR
    / "archive (2)"
    / "Lung Segmentation"
)

IMAGE_DIR = DATASET_ROOT / "CXR_png"
MASK_DIR = DATASET_ROOT / "masks"

OUTPUT_DIR = SEGMENTATION_DATA_DIR

MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_images(folder):
    """Find supported image files directly inside a folder."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_grayscale(path):
    """Load an image as a grayscale NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def resize_image(image, width, height, is_mask=False):
    """
    Resize an image or mask.

    Bilinear interpolation is used for X-rays.
    Nearest-neighbor interpolation is used for masks to preserve
    their discrete pixel values.
    """
    pil_image = Image.fromarray(image)

    if is_mask:
        interpolation = Image.Resampling.NEAREST
    else:
        interpolation = Image.Resampling.BILINEAR

    resized = pil_image.resize(
        (width, height),
        resample=interpolation,
    )

    return np.asarray(resized).copy()


def write_manifest(rows):
    """Write image/mask paths and split assignments to CSV."""
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "original_image",
                "original_mask",
                "split",
                "processed_image",
                "processed_mask",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_processed_pair(image, mask, split, pair_id):
    """Save one resized X-ray and its binary mask."""
    image_output_dir = OUTPUT_DIR / split / "images"
    mask_output_dir = OUTPUT_DIR / split / "masks"

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_output_path = (
        image_output_dir / f"{pair_id}.png"
    )

    mask_output_path = (
        mask_output_dir / f"{pair_id}_mask.png"
    )

    # Store the resized grayscale X-ray.
    Image.fromarray(image).save(image_output_path)

    # Convert mask to binary: background=0, lung=255.
    binary_mask = np.where(mask > 0, 255, 0).astype(
        np.uint8
    )

    Image.fromarray(binary_mask).save(mask_output_path)

    return image_output_path, mask_output_path


# ============================================================
# MAIN PROCESS
# ============================================================

def main():
    print("=" * 65)
    print("PNEUMONIA CDSS — SEGMENTATION DATA PREPARATION")
    print("=" * 65)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {MASK_DIR}"
        )

    if not np.isclose(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Train, validation, and test ratios must total 1.0."
        )

    # Match mask filenames to image stems.
    mask_by_stem = {}

    for mask_path in find_images(MASK_DIR):
        mask_stem = mask_path.stem.removesuffix("_mask")

        if mask_stem in mask_by_stem:
            raise ValueError(
                f"Multiple masks found for {mask_stem}"
            )

        mask_by_stem[mask_stem] = mask_path

    pairs = []

    for image_path in find_images(IMAGE_DIR):
        if image_path.stem in mask_by_stem:
            pairs.append(
                (
                    image_path,
                    mask_by_stem[image_path.stem],
                )
            )

    pairs.sort(key=lambda pair: pair[0].name)

    if not pairs:
        raise RuntimeError(
            "No matching image-mask pairs were found."
        )

    print(f"Matched image-mask pairs: {len(pairs)}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Reproducibly shuffle matched pairs before splitting.
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    total = len(pairs)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[
        train_count:train_count + val_count
    ]
    test_pairs = pairs[
        train_count + val_count:
    ]

    splits = [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]

    print("\nSplit counts:")
    for split_name, split_pairs in splits:
        print(f"  {split_name}: {len(split_pairs)}")

    manifest_rows = []
    processed_count = 0

    for split_name, split_pairs in splits:
        print(f"\nProcessing {split_name} set...")

        for image_path, mask_path in split_pairs:
            image = load_grayscale(image_path)
            mask = load_grayscale(mask_path)

            # Resize image and mask independently to the same target.
            image = resize_image(
                image,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                is_mask=False,
            )

            mask = resize_image(
                mask,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                is_mask=True,
            )

            pair_id = image_path.stem

            processed_image_path, processed_mask_path = (
                save_processed_pair(
                    image,
                    mask,
                    split_name,
                    pair_id,
                )
            )

            manifest_rows.append({
                "pair_id": pair_id,
                "original_image": str(
                    image_path.relative_to(RAW_DATA_DIR)
                ),
                "original_mask": str(
                    mask_path.relative_to(RAW_DATA_DIR)
                ),
                "split": split_name,
                "processed_image": str(
                    processed_image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "processed_mask": str(
                    processed_mask_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            })

            processed_count += 1

    write_manifest(manifest_rows)

    print("\n" + "=" * 65)
    print("SEGMENTATION DATA PREPARATION COMPLETE")
    print("=" * 65)
    print(f"Total matched pairs: {total}")
    print(f"Total processed pairs: {processed_count}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Testing pairs: {len(test_pairs)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Raw dataset files were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()



# to run  python scripts/02_prepare_segmentation_data.py