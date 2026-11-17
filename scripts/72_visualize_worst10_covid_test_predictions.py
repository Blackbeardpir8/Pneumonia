
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

COMPARISON_PATH = (
    PROJECT_ROOT /
    "results/segmentation/covid_domain_final_test/per_image_comparison.csv"
)

MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_covid_finetuned_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results/segmentation/covid_domain_final_test/error_analysis"
)

THRESHOLD = 0.30
TOP_N = 10


def load_grayscale(path):

    image = tf.io.decode_png(
        tf.io.read_file(str(path)),
        channels=1,
    )

    return (
        tf.cast(image, tf.float32) / 255.0
    ).numpy()[:, :, 0]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)

    # Select the worst-performing fine-tuned predictions.
    worst = (
        comparison
        .nsmallest(TOP_N, "finetuned_dice")
        .reset_index(drop=True)
    )

    # Join with manifest to retrieve image and mask paths.
    worst = worst.merge(
        manifest[
            [
                "original_file_name",
                "split",
                "class_name",
                "processed_image_path",
                "processed_mask_path",
            ]
        ],
        on=["original_file_name", "class_name"],
        how="left",
        validate="one_to_one",
    )

    if len(worst) != TOP_N:
        raise ValueError(
            f"Expected {TOP_N} images, found {len(worst)}"
        )

    if not (worst["split"] == "test").all():
        raise ValueError("Some selected images are not from test split.")

    print("Selected worst-performing test images:")
    print(
        worst[
            ["class_name", "original_file_name", "finetuned_dice"]
        ].to_string(index=False)
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    fig, axes = plt.subplots(
        TOP_N,
        4,
        figsize=(16, TOP_N * 3.5),
    )

    column_titles = [
        "Original X-ray",
        "Ground-truth mask",
        "Predicted mask",
        "Overlay: Green = GT, Red = Prediction",
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11)

    for index, row in worst.iterrows():

        image = load_grayscale(
            Path(row["processed_image_path"])
        )

        true_mask = load_grayscale(
            Path(row["processed_mask_path"])
        )

        true_mask = (true_mask >= 0.5).astype(np.float32)

        prediction = model.predict(
            image[np.newaxis, :, :, np.newaxis],
            verbose=0,
        )[0, :, :, 0]

        predicted_mask = (
            prediction >= THRESHOLD
        ).astype(np.float32)

        dice = float(row["finetuned_dice"])

        # Original X-ray
        axes[index, 0].imshow(image, cmap="gray")
        axes[index, 0].set_ylabel(
            f"{row['class_name']}\n{row['original_file_name']}\nDice: {dice:.4f}",
            fontsize=8,
        )

        # Ground-truth mask
        axes[index, 1].imshow(true_mask, cmap="gray")

        # Predicted mask
        axes[index, 2].imshow(predicted_mask, cmap="gray")

        # Overlay
        axes[index, 3].imshow(image, cmap="gray")
        axes[index, 3].imshow(
            true_mask,
            cmap="Greens",
            alpha=0.35,
        )
        axes[index, 3].imshow(
            predicted_mask,
            cmap="Reds",
            alpha=0.35,
        )

        for col in range(4):
            axes[index, col].axis("off")

    fig.suptitle(
        "Worst 10 COVID-domain U-Net Test Predictions",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = OUTPUT_DIR / "worst10_contact_sheet.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    worst.to_csv(
        OUTPUT_DIR / "worst10_image_details.csv",
        index=False,
    )

    print("\nContact sheet saved:", output_path)
    print("Image details saved:", OUTPUT_DIR / "worst10_image_details.csv")


if __name__ == "__main__":
    main()

## to run this script, use the following command:
# python scripts/72_visualize_worst10_covid_test_predictions.py
