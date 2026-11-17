
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from aiml.config import SAVED_MODELS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data/processed/segmentation_covid/manifest.csv"
)

MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_covid_finetuned_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results/segmentation/covid_domain_final_test/failure_inspection"
)

IMAGE_NAME = "COVID-698.png"
THRESHOLD = 0.30


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    row = df[
        (df["original_file_name"] == IMAGE_NAME) &
        (df["split"] == "test")
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one test record for {IMAGE_NAME}"
        )

    row = row.iloc[0]

    image_path = Path(row["processed_image_path"])
    mask_path = Path(row["processed_mask_path"])

    image = tf.io.decode_png(
        tf.io.read_file(str(image_path)),
        channels=1,
    )

    image = tf.cast(image, tf.float32) / 255.0

    true_mask = tf.io.decode_png(
        tf.io.read_file(str(mask_path)),
        channels=1,
    )

    true_mask = tf.cast(true_mask >= 128, tf.float32)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    prediction = model.predict(
        tf.expand_dims(image, axis=0),
        verbose=0,
    )[0, :, :, 0]

    predicted_mask = (
        prediction >= THRESHOLD
    ).astype(np.float32)

    image = image.numpy()[:, :, 0]
    true_mask = true_mask.numpy()[:, :, 0]

    intersection = np.sum(true_mask * predicted_mask)

    dice = (
        2 * intersection + 1e-6
    ) / (
        np.sum(true_mask) +
        np.sum(predicted_mask) + 1e-6
    )

    iou = (
        intersection + 1e-6
    ) / (
        np.sum(true_mask) +
        np.sum(predicted_mask) -
        intersection + 1e-6
    )

    # Save individual masks.
    plt.imsave(
        OUTPUT_DIR / "original_xray.png",
        image,
        cmap="gray",
    )

    plt.imsave(
        OUTPUT_DIR / "ground_truth_mask.png",
        true_mask,
        cmap="gray",
    )

    plt.imsave(
        OUTPUT_DIR / "predicted_mask.png",
        predicted_mask,
        cmap="gray",
    )

    # Create comparison visualization.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original X-ray")

    axes[0, 1].imshow(true_mask, cmap="gray")
    axes[0, 1].set_title("Ground-truth lung mask")

    axes[1, 0].imshow(predicted_mask, cmap="gray")
    axes[1, 0].set_title("Fine-tuned U-Net prediction")

    axes[1, 1].imshow(image, cmap="gray")
    axes[1, 1].imshow(
        true_mask,
        cmap="Greens",
        alpha=0.35,
    )
    axes[1, 1].imshow(
        predicted_mask,
        cmap="Reds",
        alpha=0.35,
    )
    axes[1, 1].set_title(
        "Overlay: Green = Ground truth, Red = Prediction"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{IMAGE_NAME} | Dice: {dice:.4f} | IoU: {iou:.4f}"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "failure_comparison.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("\nIMAGE:", IMAGE_NAME)
    print("Split:", row["split"])
    print("Class:", row["class_name"])
    print("Threshold:", THRESHOLD)
    print("Ground-truth mask area:", round(float(true_mask.mean() * 100), 3), "%")
    print("Predicted mask area:", round(float(predicted_mask.mean() * 100), 3), "%")
    print("Dice:", round(float(dice), 6))
    print("IoU:", round(float(iou), 6))
    print("\nSaved visualization:", output_path)


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/71_inspect_covid_segmentation_failure.py