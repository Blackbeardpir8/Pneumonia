from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from aiml.config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    SEGMENTATION_DATA_DIR,
    UNET_MODEL_PATH,
    SEGMENTATION_RESULTS_DIR,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(PROJECT_ROOT)

VAL_IMAGES_DIR = SEGMENTATION_DATA_DIR / "val" / "images"
VAL_MASKS_DIR = SEGMENTATION_DATA_DIR / "val" / "masks"

OUTPUT_DIR = SEGMENTATION_RESULTS_DIR / "prediction_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load best model
# --------------------------------------------------

model_path = Path(UNET_MODEL_PATH)

if not model_path.exists():
    raise FileNotFoundError(
        f"Best model not found: {model_path}"
    )

print("Loading model:", model_path)

model = tf.keras.models.load_model(
    model_path,
    compile=False,
)

print("Model loaded successfully.")


# --------------------------------------------------
# Match validation images and masks
# --------------------------------------------------

image_paths = sorted(VAL_IMAGES_DIR.glob("*.png"))
mask_paths = sorted(VAL_MASKS_DIR.glob("*.png"))

mask_by_name = {
    path.stem.removesuffix("_mask") + path.suffix: path
    for path in mask_paths
}

matched_pairs = [
    (image_path, mask_by_name[image_path.name])
    for image_path in image_paths
    if image_path.name in mask_by_name
]

if not matched_pairs:
    raise RuntimeError(
        "No matching validation image-mask pairs found."
    )

print("Validation pairs:", len(matched_pairs))


# --------------------------------------------------
# Select six samples across validation set
# --------------------------------------------------

sample_indices = np.linspace(
    0,
    len(matched_pairs) - 1,
    num=min(6, len(matched_pairs)),
    dtype=int,
)

for sample_number, index in enumerate(sample_indices, start=1):

    image_path, mask_path = matched_pairs[index]

    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE, IMAGE_SIZE, 1],
    )

    mask_bytes = tf.io.read_file(str(mask_path))
    true_mask = tf.io.decode_image(
        mask_bytes,
        channels=1,
        expand_animations=False,
    )

    true_mask = tf.cast(true_mask, tf.float32) / 255.0
    true_mask = tf.cast(true_mask >= 0.5, tf.float32)

    prediction = model.predict(
        tf.expand_dims(image, axis=0),
        verbose=0,
    )[0]

    print(
    f"{image_path.name} probability statistics: "
    f"min={prediction.min():.4f}, "
    f"mean={prediction.mean():.4f}, "
    f"max={prediction.max():.4f}, "
    f"p25={np.percentile(prediction, 25):.4f}, "
    f"p50={np.percentile(prediction, 50):.4f}, "
    f"p75={np.percentile(prediction, 75):.4f}"
)

    predicted_mask = (prediction >= 0.3).astype(np.uint8)

    true_mask_np = true_mask.numpy().squeeze()
    predicted_mask_np = predicted_mask.squeeze()

    intersection = np.sum(
        true_mask_np * predicted_mask_np
    )

    denominator = (
        np.sum(true_mask_np)
        + np.sum(predicted_mask_np)
    )

    dice = (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )

    print(
        f"{image_path.name}: "
        f"Dice={dice:.4f}, "
        f"True foreground={true_mask_np.mean():.4f}, "
        f"Predicted foreground={predicted_mask_np.mean():.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(
        image.numpy().squeeze(),
        cmap="gray",
    )
    axes[0].set_title("Original X-ray")

    axes[1].imshow(
        true_mask_np,
        cmap="gray",
    )
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(
        predicted_mask_np,
        cmap="gray",
    )
    axes[2].set_title(f"Predicted mask\nDice={dice:.3f}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(image_path.name)
    fig.tight_layout()

    output_path = OUTPUT_DIR / (
        f"sample_{sample_number}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

print("\nVisualization completed.")
print("Output directory:", OUTPUT_DIR)


# to run  python scripts/04_visualize_segmentation_predictions.py