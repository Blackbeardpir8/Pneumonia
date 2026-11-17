"""
Visually inspect chest X-rays and their corresponding lung masks.

This script:
1. Loads sample X-rays from the COVID-19 Radiography dataset.
2. Loads their corresponding masks.
3. Creates an overlay to inspect mask alignment.
4. Saves visualizations under results/dataset_audit/mask_samples/.

IMPORTANT:
- Raw dataset files are never modified.
- This script does not train a model.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# 1. LOAD PROJECT CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import RAW_DATA_DIR, RESULTS_DIR


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

DATASET_DIR = (
    RAW_DATA_DIR
    / "archive"
    / "COVID-19_Radiography_Dataset"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "dataset_audit"
    / "mask_samples"
)

CATEGORIES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
]

SAMPLES_PER_CATEGORY = 2

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def find_images(folder):
    """Find supported image files directly inside a folder."""
    if not folder.is_dir():
        return []

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


# ------------------------------------------------------------
# 4. VISUALIZE IMAGE, MASK, AND OVERLAY
# ------------------------------------------------------------

def visualize_sample(category, image_path, mask_path):
    xray = load_grayscale(image_path)
    mask = load_grayscale(mask_path)

    print(
        f"Checking {image_path.name}: "
        f"X-ray={xray.shape}, Mask={mask.shape}"
    )

    # Resize the mask to match the X-ray dimensions for visualization.
    # NEAREST interpolation preserves the mask's discrete pixel values.
    if xray.shape != mask.shape:
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize(
            (xray.shape[1], xray.shape[0]),
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(mask_image).copy()

        print(f"Resized mask for visualization: {mask.shape}")

    # Convert the mask into a binary mask.
    binary_mask = mask > 0

    # Create an RGB copy of the original X-ray.
    overlay = np.stack(
        [xray, xray, xray],
        axis=-1,
    ).astype(np.float32)

    # Highlight the mask area in red.
    overlay[binary_mask] = (
        0.5 * overlay[binary_mask]
        + 0.5 * np.array([255, 0, 0])
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Display the X-ray, mask, and overlay side by side.
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(xray, cmap="gray")
    axes[0].set_title("Original Chest X-ray")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask (Resized for Visualization)")

    axes[2].imshow(overlay)
    axes[2].set_title("Mask Overlay (Red)")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{category} — {image_path.name}",
        fontsize=12,
    )
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"{category.replace(' ', '_')}_{image_path.stem}.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved visualization: {output_path}")


# ------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("PNEUMONIA CDSS — SEGMENTATION MASK VISUAL INSPECTION")
    print("=" * 65)

    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    total = 0

    for category in CATEGORIES:
        image_dir = DATASET_DIR / category / "images"
        mask_dir = DATASET_DIR / category / "masks"

        images = find_images(image_dir)
        masks = find_images(mask_dir)

        # Match files by filename stem.
        mask_by_stem = {
            path.stem: path
            for path in masks
        }

        matched = [
            (image_path, mask_by_stem[image_path.stem])
            for image_path in images
            if image_path.stem in mask_by_stem
        ]

        print(f"\nCategory: {category}")
        print(f"Images: {len(images)}")
        print(f"Masks:  {len(masks)}")
        print(f"Matched pairs: {len(matched)}")

        if not matched:
            print("No matching image-mask pairs found.")
            continue

        for image_path, mask_path in matched[:SAMPLES_PER_CATEGORY]:
            visualize_sample(
                category,
                image_path,
                mask_path,
            )
            total += 1

    print("\n" + "=" * 65)
    print(f"Visualizations created: {total}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Raw dataset files were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/01b_visualize_segmentation_masks.py