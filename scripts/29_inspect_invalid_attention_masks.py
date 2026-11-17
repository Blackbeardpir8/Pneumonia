
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_gradcam_analysis"
    / "validation_gradcam_per_image.csv"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification/validation_gradcam_analysis"
    / "invalid_mask_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
THRESHOLD = 0.30


# =========================================================
# IMAGE LOADING
# =========================================================

def load_rgb_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    return tf.cast(image, tf.float32).numpy()


def load_grayscale_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image.numpy()


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("INSPECTING INVALID U-NET ATTENTION MASKS")
    print("=" * 65)

    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError(ANALYSIS_CSV)

    df = pd.read_csv(ANALYSIS_CSV)

    required_columns = [
        "processed_path",
        "actual_class",
        "predicted_class",
        "source_dataset",
        "lung_mask_area_percent",
        "attention_analysis_valid",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    # Select invalid attention records.
    invalid_df = df[
        ~df["attention_analysis_valid"].astype(bool)
    ].copy()

    print("\nInvalid records found:", len(invalid_df))

    if invalid_df.empty:
        print("No invalid records found.")
        return

    print("\nLoading balanced U-Net...")

    unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    inspection_results = []

    # =====================================================
    # INSPECT EACH INVALID IMAGE
    # =====================================================

    for index, (_, row) in enumerate(
        invalid_df.iterrows(),
        start=1
    ):

        image_path = Path(
            str(row["processed_path"])
        )

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        print(f"\n[{index}/{len(invalid_df)}]")
        print("Class:", row["actual_class"])
        print("Source:", row["source_dataset"])
        print("Image:", image_path)

        rgb_image = load_rgb_image(image_path)

        gray_image = load_grayscale_image(image_path)

        # Predict U-Net probability mask.
        probabilities = unet(
            gray_image[np.newaxis, ...],
            training=False
        ).numpy()[0, ..., 0]

        # Apply finalized threshold.
        binary_mask = probabilities >= THRESHOLD

        total_pixels = binary_mask.size
        lung_pixels = int(np.sum(binary_mask))

        mask_area_percent = (
            lung_pixels / total_pixels * 100
        )

        # Determine the reason attention analysis was invalid.
        if lung_pixels == 0:
            reason = "ZERO_AREA_MASK"

        elif lung_pixels == total_pixels:
            reason = "FULL_AREA_MASK"

        else:
            reason = "OTHER_INVALID_CONDITION"

        print("Predicted mask area:", f"{mask_area_percent:.2f}%")
        print("Invalid reason:", reason)

        # Create overlay.
        overlay = rgb_image / 255.0

        # Save individual visualization.
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 10)
        )

        axes[0, 0].imshow(
            rgb_image.astype(np.uint8)
        )
        axes[0, 0].set_title("Original X-ray")

        probability_plot = axes[0, 1].imshow(
            probabilities,
            cmap="jet",
            vmin=0,
            vmax=1
        )
        axes[0, 1].set_title("U-Net Probability Mask")

        fig.colorbar(
            probability_plot,
            ax=axes[0, 1],
            fraction=0.046
        )

        axes[1, 0].imshow(
            binary_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[1, 0].set_title(
            f"Binary Mask (Threshold {THRESHOLD})"
        )

        axes[1, 1].imshow(overlay)

        # Display mask in red.
        red_overlay = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 4),
            dtype=np.float32
        )

        red_overlay[..., 0] = 1.0
        red_overlay[..., 3] = (
            binary_mask.astype(np.float32) * 0.40
        )

        axes[1, 1].imshow(red_overlay)
        axes[1, 1].set_title("Predicted Lung Mask Overlay")

        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(
            f"{row['actual_class']} | {reason} | "
            f"Mask area: {mask_area_percent:.2f}%",
            fontsize=14
        )

        plt.tight_layout()

        output_path = OUTPUT_DIR / (
            f"invalid_mask_{index}_{row['actual_class']}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        inspection_results.append({
            "actual_class": row["actual_class"],
            "predicted_class": row["predicted_class"],
            "source_dataset": row["source_dataset"],
            "processed_path": str(image_path),
            "previous_mask_area_percent": row[
                "lung_mask_area_percent"
            ],
            "recalculated_mask_area_percent": mask_area_percent,
            "reason": reason,
            "minimum_probability": float(
                np.min(probabilities)
            ),
            "maximum_probability": float(
                np.max(probabilities)
            ),
            "mean_probability": float(
                np.mean(probabilities)
            ),
            "visualization": str(output_path),
        })

    # =====================================================
    # SAVE INSPECTION SUMMARY
    # =====================================================

    result_df = pd.DataFrame(inspection_results)

    result_df.to_csv(
        OUTPUT_DIR / "invalid_mask_inspection_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETED")
    print("=" * 65)

    print("\nInspection summary:")
    print(
        result_df[
            [
                "actual_class",
                "reason",
                "recalculated_mask_area_percent",
                "mean_probability",
            ]
        ].to_string(index=False)
    )

    print("\nOutput folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/29_inspect_invalid_attention_masks.py