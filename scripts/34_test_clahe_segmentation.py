
import sys
from pathlib import Path

import cv2
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

VALIDATION_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_evaluation"
    / "validation_predictions.csv"
)

INVALID_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_gradcam_analysis"
    / "invalid_mask_inspection"
    / "invalid_mask_inspection_summary.csv"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification"
    / "clahe_segmentation_experiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
CONTROL_SAMPLE_SIZE = 20
RANDOM_SEED = 42

FINAL_THRESHOLD = 0.30

SOURCE_GROUPS = [
    "covid19_radiography",
    "bacterial_viral_dataset",
]

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# =========================================================
# LOAD ORIGINAL IMAGE
# =========================================================

def load_original_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    return image.numpy().squeeze()


# =========================================================
# CLAHE ENHANCEMENT
# =========================================================

def apply_clahe(image):

    image = np.asarray(image, dtype=np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )

    enhanced = clahe.apply(image)

    return enhanced


# =========================================================
# PREPROCESS FOR U-NET
# =========================================================

def preprocess_for_unet(image):

    image = tf.convert_to_tensor(
        image[..., np.newaxis],
        dtype=tf.float32,
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
        antialias=False,
    )

    image = image / 255.0

    return image.numpy().astype(np.float32)


# =========================================================
# CALCULATE INPUT STATISTICS
# =========================================================

def calculate_input_statistics(image):

    image = np.asarray(image, dtype=np.float32)

    return {
        "input_mean": float(np.mean(image)),
        "input_std": float(np.std(image)),
        "input_min": float(np.min(image)),
        "input_max": float(np.max(image)),
        "input_p05": float(np.percentile(image, 5)),
        "input_p95": float(np.percentile(image, 95)),
    }


# =========================================================
# CALCULATE PREDICTION STATISTICS
# =========================================================

def calculate_prediction_statistics(probability_map):

    binary_mask = probability_map >= FINAL_THRESHOLD

    return {
        "probability_mean": float(np.mean(probability_map)),
        "probability_std": float(np.std(probability_map)),
        "probability_max": float(np.max(probability_map)),
        "probability_median": float(np.median(probability_map)),
        "mask_area_percent": float(
            np.mean(binary_mask) * 100
        ),
        "mask_status": (
            "empty"
            if not np.any(binary_mask)
            else "non_empty"
        ),
    }


# =========================================================
# SELECT SAME 44 IMAGES
# =========================================================

def select_images():

    validation_df = pd.read_csv(VALIDATION_CSV)
    invalid_df = pd.read_csv(INVALID_CSV)

    validation_df["absolute_path"] = (
        validation_df["processed_path"].apply(
            lambda value: (
                Path(value)
                if Path(value).is_absolute()
                else PROJECT_ROOT / Path(value)
            )
        )
    )

    invalid_paths = set(
        invalid_df["processed_path"].apply(
            lambda value: str(
                Path(value)
                if Path(value).is_absolute()
                else PROJECT_ROOT / Path(value)
            )
        )
    )

    failed_df = validation_df[
        validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    if len(failed_df) != len(invalid_paths):
        raise RuntimeError(
            "Failed image matching count is incorrect."
        )

    if failed_df.empty:
        raise RuntimeError("No failed images found.")

    failed_df["comparison_group"] = "failed"

    control_candidates = validation_df[
        ~validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    controls = []

    for source in SOURCE_GROUPS:

        source_df = control_candidates[
            control_candidates["source_dataset"].astype(str)
            == source
        ]

        if source_df.empty:
            raise RuntimeError(
                f"No successful controls found for {source}"
            )

        sample_count = min(
            CONTROL_SAMPLE_SIZE,
            len(source_df),
        )

        selected = source_df.sample(
            n=sample_count,
            random_state=RANDOM_SEED,
        ).copy()

        selected["comparison_group"] = (
            f"successful_{source}"
        )

        controls.append(selected)

    return pd.concat(
        [failed_df] + controls,
        ignore_index=True,
    )


# =========================================================
# CREATE FAILED IMAGE PREVIEW
# =========================================================

def create_failed_preview(preview_data):

    if not preview_data:
        return

    fig, axes = plt.subplots(
        len(preview_data),
        4,
        figsize=(16, len(preview_data) * 4),
        squeeze=False,
    )

    for index, item in enumerate(preview_data):

        axes[index, 0].imshow(
            item["original"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[index, 0].set_title("Original X-ray")

        axes[index, 1].imshow(
            item["clahe"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[index, 1].set_title("CLAHE-enhanced X-ray")

        axes[index, 2].imshow(
            item["original_probability"],
            cmap="jet",
            vmin=0,
            vmax=1,
        )
        axes[index, 2].set_title(
            "Original U-Net probability\n"
            f"Max: {item['original_max']:.4f} | "
            f"Mask: {item['original_area']:.4f}%"
        )

        axes[index, 3].imshow(
            item["clahe_probability"],
            cmap="jet",
            vmin=0,
            vmax=1,
        )
        axes[index, 3].set_title(
            "CLAHE U-Net probability\n"
            f"Max: {item['clahe_max']:.4f} | "
            f"Mask: {item['clahe_area']:.4f}%"
        )

        for axis in axes[index]:
            axis.axis("off")

        axes[index, 0].set_ylabel(
            f"{item['actual_class']}\n"
            f"{item['image_name']}",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "failed_images_clahe_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CLAHE U-NET SEGMENTATION EXPERIMENT")
    print("=" * 70)

    for path in [
        VALIDATION_CSV,
        INVALID_CSV,
        UNET_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    selected_df = select_images()

    print("\nSelected images:")
    print(
        selected_df["comparison_group"]
        .value_counts()
        .to_string()
    )

    print("\nLoading balanced U-Net...")

    model = tf.keras.models.load_model(
        UNET_PATH,
        compile=False,
    )

    results = []
    failed_preview_data = []

    # =====================================================
    # PROCESS IMAGES
    # =====================================================

    for image_number, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1,
    ):

        image_path = Path(row["absolute_path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        original = load_original_image(image_path)

        clahe_image = apply_clahe(original)

        # Both inputs use the same resize and normalization.
        original_input = preprocess_for_unet(original)
        clahe_input = preprocess_for_unet(clahe_image)

        # Predict independently.
        original_probability = model(
            original_input[None, ...],
            training=False,
        ).numpy()[0, ..., 0]

        clahe_probability = model(
            clahe_input[None, ...],
            training=False,
        ).numpy()[0, ..., 0]

        # Save results for both conditions.
        for condition, input_image, probability in [
            (
                "original",
                original_input,
                original_probability,
            ),
            (
                "clahe",
                clahe_input,
                clahe_probability,
            ),
        ]:

            input_stats = calculate_input_statistics(
                input_image
            )

            prediction_stats = (
                calculate_prediction_statistics(
                    probability
                )
            )

            results.append({
                "comparison_group": row["comparison_group"],
                "source_dataset": row["source_dataset"],
                "actual_class": row.get("actual_class"),
                "predicted_class": row.get("predicted_class"),
                "condition": condition,
                "processed_path": str(image_path),
                **input_stats,
                **prediction_stats,
            })

        # Keep failed images for visual comparison.
        if row["comparison_group"] == "failed":

            original_stats = calculate_prediction_statistics(
                original_probability
            )

            clahe_stats = calculate_prediction_statistics(
                clahe_probability
            )

            failed_preview_data.append({
                "image_name": image_path.name,
                "actual_class": row.get("actual_class"),
                "original": original,
                "clahe": clahe_image,
                "original_probability": original_probability,
                "clahe_probability": clahe_probability,
                "original_max": original_stats["probability_max"],
                "clahe_max": clahe_stats["probability_max"],
                "original_area": original_stats["mask_area_percent"],
                "clahe_area": clahe_stats["mask_area_percent"],
            })

        print(
            f"Processed image {image_number}/{len(selected_df)}"
        )

    # =====================================================
    # SAVE PER-IMAGE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "clahe_per_image_results.csv",
        index=False,
    )

    # =====================================================
    # SUMMARY BY GROUP AND CONDITION
    # =====================================================

    summary = (
        results_df.groupby(
            ["comparison_group", "condition"],
            dropna=False,
        )
        .agg(
            image_count=("processed_path", "nunique"),
            mean_input_intensity=("input_mean", "mean"),
            mean_input_contrast=("input_std", "mean"),
            mean_probability=("probability_mean", "mean"),
            mean_max_probability=("probability_max", "mean"),
            mean_mask_area=("mask_area_percent", "mean"),
            empty_mask_count=(
                "mask_status",
                lambda values: (values == "empty").sum(),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "clahe_group_summary.csv",
        index=False,
    )

    # =====================================================
    # FAILED IMAGE COMPARISON
    # =====================================================

    failed_results = results_df[
        results_df["comparison_group"] == "failed"
    ]

    failed_comparison = failed_results.pivot(
        index="processed_path",
        columns="condition",
        values=[
            "probability_mean",
            "probability_max",
            "mask_area_percent",
        ],
    )

    failed_comparison.to_csv(
        OUTPUT_DIR / "failed_image_original_vs_clahe.csv"
    )

    # =====================================================
    # CREATE VISUALIZATION
    # =====================================================

    create_failed_preview(failed_preview_data)

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print("CLAHE EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nGroup summary:")
    print(summary.to_string(index=False))

    print("\nFailed image comparison:")
    print(
        failed_results[
            [
                "actual_class",
                "condition",
                "input_std",
                "probability_mean",
                "probability_max",
                "mask_area_percent",
                "mask_status",
            ]
        ].to_string(index=False)
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- clahe_per_image_results.csv")
    print("- clahe_group_summary.csv")
    print("- failed_image_original_vs_clahe.csv")
    print("- failed_images_clahe_comparison.png")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/25_evaluate_final_test_segmentation.py