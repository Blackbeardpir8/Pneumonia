
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
)

VAL_DIR = DATASET_ROOT / "Val" / "COVID-19"

IMAGE_DIR = VAL_DIR / "images"
INFECTION_MASK_DIR = VAL_DIR / "infection masks"
LUNG_MASK_DIR = VAL_DIR / "lung masks"

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "threshold_analysis"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BATCH_SIZE = 8

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

EPSILON = 1e-7


# ============================================================
# 2. FILE INDEX
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg"
}


def build_file_index(directory):

    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


# ============================================================
# 3. LOAD VALIDATION DATA
# ============================================================

def load_image_and_masks(image_path, infection_path, lung_path):

    image = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=1,
        expand_animations=False,
    )

    infection_mask = tf.io.decode_image(
        tf.io.read_file(infection_path),
        channels=1,
        expand_animations=False,
    )

    lung_mask = tf.io.decode_image(
        tf.io.read_file(lung_path),
        channels=1,
        expand_animations=False,
    )

    image.set_shape([None, None, 1])
    infection_mask.set_shape([None, None, 1])
    lung_mask.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    infection_mask = tf.image.resize(
        infection_mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    lung_mask = tf.image.resize(
        lung_mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    image = tf.cast(image, tf.float32) / 255.0

    infection_mask = tf.cast(
        tf.cast(infection_mask, tf.float32) >= 128.0,
        tf.uint8,
    )

    lung_mask = tf.cast(
        tf.cast(lung_mask, tf.float32) >= 128.0,
        tf.uint8,
    )

    return image, infection_mask, lung_mask


# ============================================================
# 4. METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.logical_and(y_true, y_pred).sum())

    fp = int(
        np.logical_and(
            np.logical_not(y_true),
            y_pred,
        ).sum()
    )

    fn = int(
        np.logical_and(
            y_true,
            np.logical_not(y_pred),
        ).sum()
    )

    gt_pixels = int(y_true.sum())
    pred_pixels = int(y_pred.sum())

    dice = (
        2.0 * tp + EPSILON
    ) / (
        2.0 * tp + fp + fn + EPSILON
    )

    iou = (
        tp + EPSILON
    ) / (
        tp + fp + fn + EPSILON
    )

    precision = (
        tp + EPSILON
    ) / (
        tp + fp + EPSILON
    )

    recall = (
        tp + EPSILON
    ) / (
        tp + fn + EPSILON
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "ground_truth_pixels": gt_pixels,
        "predicted_pixels": pred_pixels,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "predicted_to_reference_area_ratio": float(
            pred_pixels / (gt_pixels + EPSILON)
        ),
    }


# ============================================================
# 5. ANNOTATION SUBGROUP CLASSIFICATION
# ============================================================

def classify_annotation(infection_mask, lung_mask):

    infection_pixels = int(infection_mask.sum())
    lung_pixels = int(lung_mask.sum())

    if lung_pixels == 0:
        return "INVALID_EMPTY_LUNG_MASK"

    if np.array_equal(infection_mask, lung_mask):
        return "IDENTICAL_TO_LUNG"

    ratio = infection_pixels / lung_pixels * 100.0

    if ratio < 10.0:
        return "NON_IDENTICAL_SMALL"

    elif ratio <= 50.0:
        return "NON_IDENTICAL_MODERATE"

    return "NON_IDENTICAL_LARGE"


# ============================================================
# 6. SUBGROUP SUMMARY
# ============================================================

def create_summary(data):

    summary = (
        data.groupby(
            ["threshold", "annotation_group"],
            sort=False,
        )
        .agg(
            sample_count=("image", "count"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            mean_iou=("iou", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_positive_pixels=(
                "false_positive_pixels", "mean"
            ),
            mean_false_negative_pixels=(
                "false_negative_pixels", "mean"
            ),
            mean_area_ratio=(
                "predicted_to_reference_area_ratio", "mean"
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COVID INFECTION U-NET - THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    image_files = build_file_index(IMAGE_DIR)
    infection_files = build_file_index(INFECTION_MASK_DIR)
    lung_files = build_file_index(LUNG_MASK_DIR)

    common_names = (
        set(image_files)
        & set(infection_files)
        & set(lung_files)
    )

    if not common_names:
        raise ValueError("No matched validation samples found.")

    if not (
        set(image_files)
        == set(infection_files)
        == set(lung_files)
    ):
        raise ValueError(
            "Validation image, infection-mask and lung-mask "
            "filenames do not match exactly."
        )

    names = sorted(common_names)

    print("Matched validation images:", len(names))

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load each validation image only once.
    # Generate probabilities once and reuse all thresholds.
    # --------------------------------------------------------

    records = []

    for start in range(0, len(names), BATCH_SIZE):

        batch_names = names[start:start + BATCH_SIZE]

        images = []
        masks = []
        lung_masks = []

        for name in batch_names:

            image, infection_mask, lung_mask = (
                load_image_and_masks(
                    str(image_files[name]),
                    str(infection_files[name]),
                    str(lung_files[name]),
                )
            )

            images.append(image.numpy())
            masks.append(infection_mask.numpy().squeeze())
            lung_masks.append(lung_mask.numpy().squeeze())

        images = np.asarray(images, dtype=np.float32)

        masks = np.asarray(masks, dtype=np.uint8)
        lung_masks = np.asarray(lung_masks, dtype=np.uint8)

        probabilities = model(
            images,
            training=False,
        ).numpy()

        for i, name in enumerate(batch_names):

            probability = probabilities[i, :, :, 0]

            ground_truth = masks[i]
            lung_mask = lung_masks[i]

            group = classify_annotation(
                ground_truth,
                lung_mask,
            )

            for threshold in THRESHOLDS:

                predicted_mask = (
                    probability >= threshold
                )

                metrics = calculate_metrics(
                    ground_truth,
                    predicted_mask,
                )

                records.append({

                    "image": name,

                    "annotation_group": group,

                    "threshold": threshold,

                    **metrics,
                })

        print(
            f"Processed {min(start + BATCH_SIZE, len(names))}"
            f"/{len(names)} validation images"
        )

    # --------------------------------------------------------
    # SAVE PER-IMAGE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(records)

    per_image_path = (
        OUTPUT_DIR
        / "validation_threshold_per_image.csv"
    )

    results.to_csv(
        per_image_path,
        index=False,
    )

    # --------------------------------------------------------
    # GLOBAL THRESHOLD SUMMARY
    # --------------------------------------------------------

    global_summary = (
        results.groupby("threshold")
        .agg(
            sample_count=("image", "nunique"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            mean_iou=("iou", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_positive_pixels=(
                "false_positive_pixels", "mean"
            ),
            mean_false_negative_pixels=(
                "false_negative_pixels", "mean"
            ),
            mean_area_ratio=(
                "predicted_to_reference_area_ratio", "mean"
            ),
        )
        .reset_index()
    )

    global_path = (
        OUTPUT_DIR
        / "validation_threshold_global_summary.csv"
    )

    global_summary.to_csv(
        global_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUBGROUP THRESHOLD SUMMARY
    # --------------------------------------------------------

    subgroup_summary = create_summary(results)

    subgroup_path = (
        OUTPUT_DIR
        / "validation_threshold_subgroup_summary.csv"
    )

    subgroup_summary.to_csv(
        subgroup_path,
        index=False,
    )

    # --------------------------------------------------------
    # BEST VALIDATION THRESHOLD BY MEAN DICE
    # --------------------------------------------------------

    best_row = global_summary.loc[
        global_summary["mean_dice"].idxmax()
    ]

    best_threshold = float(best_row["threshold"])

    summary_json = {

        "experiment": "COVID Infection Segmentation Threshold Analysis",

        "dataset": "COVID-QU-Ex validation split",

        "validation_samples": int(len(names)),

        "thresholds": THRESHOLDS,

        "selection_metric": "Mean per-image Dice",

        "best_validation_threshold": best_threshold,

        "best_validation_mean_dice": float(
            best_row["mean_dice"]
        ),

        "best_validation_mean_iou": float(
            best_row["mean_iou"]
        ),

        "best_validation_mean_precision": float(
            best_row["mean_precision"]
        ),

        "best_validation_mean_recall": float(
            best_row["mean_recall"]
        ),

        "note": (
            "Threshold selection uses validation data only. "
            "The held-out test set is not used to select "
            "the threshold."
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "validation_threshold_analysis_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_json,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GLOBAL VALIDATION THRESHOLD RESULTS")
    print("=" * 70)

    print(
        global_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nBEST VALIDATION THRESHOLD")
    print("-" * 45)

    print("Threshold:", best_threshold)
    print(
        "Mean Dice:",
        round(float(best_row["mean_dice"]), 4),
    )

    print("\nSUBGROUP THRESHOLD RESULTS")
    print("-" * 100)

    print(
        subgroup_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved files:")
    print(per_image_path)
    print(global_path)
    print(subgroup_path)
    print(json_path)

    print("\nTHRESHOLD SENSITIVITY ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/98_analyze_infection_threshold_sensitivity.py