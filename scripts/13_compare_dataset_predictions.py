
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from aiml.config import (
    SEGMENTATION_DATA_DIR,
    UNET_MODEL_PATH,
)

THRESHOLD = 0.30

SAMPLES = [
    "CHNCXR_0317_0.png",
    "MCUCXR_0041_0.png",
]

OUTPUT_DIR = Path("results/segmentation/error_analysis")


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L"),
            dtype=np.float32,
        )


def calculate_dice(true_mask, predicted_mask):
    intersection = np.logical_and(
        true_mask, predicted_mask
    ).sum()

    denominator = true_mask.sum() + predicted_mask.sum()

    return (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )


def main():
    data_root = Path(SEGMENTATION_DATA_DIR)

    image_dir = data_root / "val" / "images"
    mask_dir = data_root / "val" / "masks"

    model_path = Path(UNET_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print("Loading trained U-Net...")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    fig, axes = plt.subplots(
        len(SAMPLES),
        4,
        figsize=(16, 8),
    )

    for row, image_name in enumerate(SAMPLES):
        image_path = image_dir / image_name

        mask_path = (
            mask_dir /
            f"{Path(image_name).stem}_mask.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = load_image(image_path)
        true_mask = load_image(mask_path) >= 128

        probabilities = model.predict(
            image[None, ..., None] / 255.0,
            verbose=0,
        )[0, ..., 0]

        predicted_mask = probabilities >= THRESHOLD

        dice = calculate_dice(
            true_mask,
            predicted_mask,
        )

        # Original X-ray
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title("Original X-ray")

        # Ground-truth mask
        axes[row, 1].imshow(
            true_mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title("Ground-truth mask")

        # Predicted mask
        axes[row, 2].imshow(
            predicted_mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title(
            f"Predicted mask\nDice = {dice:.4f}"
        )

        # Overlay
        overlay = np.stack(
            [image / 255.0] * 3,
            axis=-1,
        )

        overlay[true_mask & ~predicted_mask] = [
            0.0, 1.0, 0.0
        ]

        overlay[predicted_mask & ~true_mask] = [
            1.0, 0.0, 0.0
        ]

        overlay[true_mask & predicted_mask] = [
            1.0, 1.0, 0.0
        ]

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay")

        print(
            f"{image_name}: Dice={dice:.4f}, "
            f"reference area={true_mask.mean():.4f}, "
            f"predicted area={predicted_mask.mean():.4f}"
        )

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        "CHNCXR vs MCUCXR Lung Segmentation Comparison",
        fontsize=15,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR /
        "CHNCXR_vs_MCUCXR_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nVisualization saved:")
    print(output_path.resolve())
    print("Model and dataset were not modified.")


if __name__ == "__main__":
    main()


# to run script, use the following command:
# python scripts/13_compare_dataset_predictions.py