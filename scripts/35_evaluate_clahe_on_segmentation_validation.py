
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data/processed/segmentation/manifest.csv"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models/unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/segmentation/clahe_validation_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

THRESHOLD = 0.30

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# =========================================================
# IMAGE LOADING
# =========================================================

def load_grayscale(path):

    image_bytes = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    return image.numpy().squeeze()


# =========================================================
# CLAHE
# =========================================================

def apply_clahe(image):

    image = np.asarray(image, dtype=np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )

    return clahe.apply(image)


# =========================================================
# PREPROCESSING
# =========================================================

def prepare_input(image):

    image = image.astype(np.float32) / 255.0

    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    return image[..., np.newaxis]


# =========================================================
# SEGMENTATION METRICS
# =========================================================

def calculate_metrics(true_mask, predicted_mask):

    true_mask = true_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    tp = int(np.sum(true_mask & predicted_mask))
    fp = int(np.sum(~true_mask & predicted_mask))
    fn = int(np.sum(true_mask & ~predicted_mask))

    dice_denominator = (
        2 * tp + fp + fn
    )

    union = tp + fp + fn

    dice = (
        (2 * tp / dice_denominator)
        if dice_denominator > 0
        else 1.0
    )

    iou = (
        tp / union
        if union > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "true_mask_area_percent": float(
            np.mean(true_mask) * 100
        ),
        "predicted_mask_area_percent": float(
            np.mean(predicted_mask) * 100
        ),
        "empty_prediction": int(
            not np.any(predicted_mask)
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


# =========================================================
# EVALUATE ONE CONDITION
# =========================================================

def evaluate_condition(model, image, true_mask):

    model_input = prepare_input(image)

    probability_map = model(
        model_input[np.newaxis, ...],
        training=False,
    ).numpy()[0, ..., 0]

    predicted_mask = probability_map >= THRESHOLD

    metrics = calculate_metrics(
        true_mask,
        predicted_mask,
    )

    metrics.update({
        "probability_mean": float(
            np.mean(probability_map)
        ),
        "probability_max": float(
            np.max(probability_map)
        ),
    })

    return metrics


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CLAHE VALIDATION EVALUATION")
    print("=" * 70)

    for path in [MANIFEST_PATH, UNET_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "pair_id",
        "split",
        "processed_image",
        "processed_mask",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    # Use only the existing validation split.
    validation_df = manifest[
        manifest["split"].astype(str).str.lower() == "val"
    ].copy()

    if validation_df.empty:
        raise RuntimeError(
            "No validation records found in manifest."
        )

    print("\nValidation image count:", len(validation_df))

    print("\nLoading balanced U-Net...")

    model = tf.keras.models.load_model(
        UNET_PATH,
        compile=False,
    )

    results = []

    # =====================================================
    # PROCESS VALIDATION IMAGES
    # =====================================================

    for index, (_, row) in enumerate(
        validation_df.iterrows(),
        start=1,
    ):

        image_path = (
            PROJECT_ROOT / Path(row["processed_image"])
        )

        mask_path = (
            PROJECT_ROOT / Path(row["processed_mask"])
        )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        original_image = load_grayscale(image_path)

        original_mask = load_grayscale(mask_path)

        if original_image.shape != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ):
            raise ValueError(
                f"Unexpected image dimensions: {image_path}"
            )

        if original_mask.shape != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ):
            raise ValueError(
                f"Unexpected mask dimensions: {mask_path}"
            )

        # Same ground-truth mask for both conditions.
        true_mask = original_mask >= 128

        if not np.any(true_mask):
            raise ValueError(
                f"Empty ground-truth mask: {mask_path}"
            )

        # Condition 1: original image.
        original_metrics = evaluate_condition(
            model,
            original_image,
            true_mask,
        )

        # Condition 2: CLAHE-enhanced image.
        clahe_image = apply_clahe(original_image)

        clahe_metrics = evaluate_condition(
            model,
            clahe_image,
            true_mask,
        )

        # Save both conditions.
        for condition, metrics in [
            ("original", original_metrics),
            ("clahe", clahe_metrics),
        ]:

            results.append({
                "pair_id": row["pair_id"],
                "condition": condition,
                "processed_image": str(image_path),
                "processed_mask": str(mask_path),
                **metrics,
            })

        print(
            f"Processed {index}/{len(validation_df)}"
        )

    # =====================================================
    # SAVE PER-IMAGE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "validation_per_image_results.csv",
        index=False,
    )

    # =====================================================
    # SUMMARY BY CONDITION
    # =====================================================

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "true_mask_area_percent",
        "predicted_mask_area_percent",
        "probability_mean",
        "probability_max",
    ]

    summary = (
        results_df.groupby("condition")
        .agg(
            image_count=("pair_id", "nunique"),
            **{
                f"mean_{column}": (column, "mean")
                for column in metric_columns
            },
            empty_prediction_count=(
                "empty_prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False,
    )

    # =====================================================
    # PAIRED COMPARISON
    # =====================================================

    original_df = (
        results_df[
            results_df["condition"] == "original"
        ]
        .set_index("pair_id")
    )

    clahe_df = (
        results_df[
            results_df["condition"] == "clahe"
        ]
        .set_index("pair_id")
    )

    comparison_metrics = [
        "dice",
        "iou",
        "precision",
        "recall",
        "predicted_mask_area_percent",
        "probability_mean",
        "probability_max",
    ]

    comparison = pd.DataFrame(index=original_df.index)

    for metric in comparison_metrics:

        comparison[f"{metric}_original"] = (
            original_df[metric]
        )

        comparison[f"{metric}_clahe"] = (
            clahe_df[metric]
        )

        comparison[f"{metric}_difference"] = (
            clahe_df[metric] - original_df[metric]
        )

    comparison = comparison.reset_index()

    comparison.to_csv(
        OUTPUT_DIR / "paired_comparison.csv",
        index=False,
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print("CLAHE VALIDATION EVALUATION COMPLETED")
    print("=" * 70)

    display_columns = [
        "condition",
        "image_count",
        "mean_dice",
        "mean_iou",
        "mean_precision",
        "mean_recall",
        "mean_predicted_mask_area_percent",
        "empty_prediction_count",
    ]

    print("\nValidation summary:")

    print(
        summary[display_columns].to_string(index=False)
    )

    print("\nMean metric differences (CLAHE - Original):")

    for metric in [
        "dice",
        "iou",
        "precision",
        "recall",
    ]:

        difference = (
            clahe_df[metric].mean()
            - original_df[metric].mean()
        )

        print(f"{metric}: {difference:+.6f}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- validation_per_image_results.csv")
    print("- validation_summary.csv")
    print("- paired_comparison.csv")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/35_evaluate_clahe_on_segmentation_validation.py
