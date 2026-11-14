
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from aiml.config import SEGMENTATION_DATA_DIR, UNET_MODEL_PATH

THRESHOLD = 0.30


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
        if denominator > 0 else 1.0
    )


def main():
    root = Path(SEGMENTATION_DATA_DIR)
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"

    model = tf.keras.models.load_model(
        UNET_MODEL_PATH,
        compile=False,
    )

    rows = []

    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = (
            mask_dir /
            f"{image_path.stem}_mask.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = load_image(image_path) / 255.0
        true_mask = load_image(mask_path) >= 128

        probabilities = model.predict(
            image[None, ..., None],
            verbose=0,
        )[0, ..., 0]

        predicted_mask = probabilities >= THRESHOLD

        dice = calculate_dice(
            true_mask,
            predicted_mask,
        )

        rows.append({
            "image": image_path.name,
            "source": image_path.name.split("_")[0],
            "dice": dice,
        })

    df = pd.DataFrame(rows)

    print("\nTRAINING PERFORMANCE BY SOURCE")
    print("=" * 55)

    print(
        df.groupby("source")["dice"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(4)
        .to_string()
    )

    print("\nTotal evaluated:", len(df))
    print("Threshold:", THRESHOLD)
    print("No model or dataset files were modified.")


if __name__ == "__main__":
    main()


# to run