import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import IMAGE_HEIGHT, IMAGE_WIDTH

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

CSV_PATH = (
    PROJECT_ROOT
    / "results/classification/validation_evaluation"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification/lung_ablation_preview"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.30

print("Loading validation records...")
df = pd.read_csv(CSV_PATH)

# Select five images from different classes.
samples = (
    df.groupby("actual_class", sort=False)
    .head(2)
    .groupby("actual_class", sort=False)
    .head(1)
    .head(5)
)

print("Selected samples:", len(samples))

print("Loading balanced U-Net...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def load_grayscale(path):
    image = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False
    )

    image.set_shape([None, None, 1])

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH]
    )

    return image.numpy()


for index, row in enumerate(samples.itertuples(), start=1):

    image_path = Path(row.processed_path)

    image = load_grayscale(image_path)

    probabilities = model.predict(
        image[None, ...],
        verbose=0
    )[0, ..., 0]

    mask = probabilities >= THRESHOLD

    # RGB display image for classification-compatible visualization.
    original_rgb = np.repeat(image, 3, axis=-1)

    lung_only = original_rgb * mask[..., None]

    background_only = original_rgb * (~mask)[..., None]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Lung Mask")

    axes[2].imshow(lung_only)
    axes[2].set_title("Lung Only")

    axes[3].imshow(background_only)
    axes[3].set_title("Background Only")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{row.actual_class} | {image_path.name}"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / f"preview_{index}_{row.actual_class}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[{index}/{len(samples)}] "
        f"{row.actual_class}: "
        f"Mask area={mask.mean():.3f}"
    )

print("\nPreview generation completed.")
print("Output folder:", OUTPUT_DIR)


# to run this script, use the following command:
# python scripts/16_preview_lung_ablation_masks.py