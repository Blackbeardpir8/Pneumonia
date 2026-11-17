
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from aiml.config import RAW_DATA_DIR, SEGMENTATION_DATA_DIR

IMAGE_NAME = "MCUCXR_0041_0.png"

RAW_ROOT = RAW_DATA_DIR / "archive (2)" / "Lung Segmentation"

ORIGINAL_IMAGE = RAW_ROOT / "CXR_png" / IMAGE_NAME
ORIGINAL_MASK = RAW_ROOT / "masks" / IMAGE_NAME

PROCESSED_IMAGE = (
    SEGMENTATION_DATA_DIR / "val" / "images" / IMAGE_NAME
)

PROCESSED_MASK = (
    SEGMENTATION_DATA_DIR
    / "val"
    / "masks"
    / f"{Path(IMAGE_NAME).stem}_mask.png"
)


def load_grayscale(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def main():
    original_image = load_grayscale(ORIGINAL_IMAGE)
    original_mask = load_grayscale(ORIGINAL_MASK) >= 128

    processed_image = load_grayscale(PROCESSED_IMAGE)
    processed_mask = load_grayscale(PROCESSED_MASK) >= 128

    overlay = np.stack(
        [processed_image] * 3,
        axis=-1,
    ).astype(np.float32) / 255.0

    # Highlight the reference lung regions in green.
    overlay[processed_mask, 1] = np.maximum(
        overlay[processed_mask, 1],
        0.85,
    )

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original X-ray")

    axes[1].imshow(original_mask, cmap="gray")
    axes[1].set_title("Original mask")

    axes[2].imshow(processed_image, cmap="gray")
    axes[2].set_title("Processed X-ray")

    axes[3].imshow(processed_mask, cmap="gray")
    axes[3].set_title("Processed mask")

    axes[4].imshow(overlay)
    axes[4].set_title("Processed X-ray + mask")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Preprocessing inspection: {IMAGE_NAME}",
        fontsize=14,
    )

    plt.tight_layout()

    output_dir = Path("results/segmentation/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "MCUCXR_0041_0_preprocessing.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {output_path.resolve()}")
    print("Original files and processed data were not modified.")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/11_visualize_preprocessing.py