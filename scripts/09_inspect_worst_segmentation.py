from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from aiml.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEGMENTATION_DATA_DIR,
    UNET_MODEL_PATH,
)

IMAGE_NAME = "MCUCXR_0041_0.png"
THRESHOLD = 0.30


def load_grayscale(path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False,
    )
    return image.numpy().squeeze()


def main():
    data_root = Path(SEGMENTATION_DATA_DIR)

    image_path = data_root / "val" / "images" / IMAGE_NAME
    mask_path = (
        data_root
        / "val"
        / "masks"
        / f"{Path(IMAGE_NAME).stem}_mask.png"
    )

    model_path = Path(UNET_MODEL_PATH)

    for path in (image_path, mask_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    image = load_grayscale(image_path).astype(np.float32) / 255.0
    true_mask = load_grayscale(mask_path) >= 128

    expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

    if image.shape != expected_shape:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    if true_mask.shape != expected_shape:
        raise ValueError(f"Unexpected mask shape: {true_mask.shape}")

    model = tf.keras.models.load_model(model_path, compile=False)

    probabilities = model.predict(
        image[None, ..., None],
        verbose=0,
    )[0, ..., 0]

    predicted_mask = probabilities >= THRESHOLD

    intersection = np.logical_and(
        true_mask, predicted_mask
    ).sum()

    denominator = true_mask.sum() + predicted_mask.sum()

    dice = (
        2.0 * intersection / denominator
        if denominator
        else 1.0
    )

    # Overlay colors:
    # Green = reference mask only
    # Red = predicted mask only
    # Yellow = overlap between the masks
    overlay = np.zeros((*true_mask.shape, 3), dtype=np.float32)

    overlay[true_mask & ~predicted_mask] = [0.0, 1.0, 0.0]
    overlay[predicted_mask & ~true_mask] = [1.0, 0.0, 0.0]
    overlay[true_mask & predicted_mask] = [1.0, 1.0, 0.0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original X-ray")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title(f"Predicted mask (threshold={THRESHOLD})")

    axes[3].imshow(overlay)
    axes[3].set_title(
        "Overlay: Green=missed, Red=extra, Yellow=overlap"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{IMAGE_NAME} | Dice={dice:.4f}",
        fontsize=14,
    )

    plt.tight_layout()

    output_dir = Path("results/segmentation/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "MCUCXR_0041_0_analysis.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Image: {IMAGE_NAME}")
    print(f"Dice: {dice:.4f}")
    print(f"Reference area: {true_mask.mean():.4f}")
    print(f"Predicted area: {predicted_mask.mean():.4f}")
    print(f"Saved visualization: {output_path.resolve()}")


if __name__ == "__main__":
    main()

# to run 