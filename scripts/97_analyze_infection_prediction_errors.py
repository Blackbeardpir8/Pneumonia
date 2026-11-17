
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

TEST_DIR = DATASET_ROOT / "Test" / "COVID-19"

IMAGE_DIR = TEST_DIR / "images"
INFECTION_MASK_DIR = TEST_DIR / "infection masks"
LUNG_MASK_DIR = TEST_DIR / "lung masks"

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

SUBGROUP_PATH = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "subgroup_analysis"
    / "test_subgroup_per_image.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "prediction_error_analysis"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
THRESHOLD = 0.5
EPSILON = 1e-7


# ============================================================
# 2. LOAD MASK
# ============================================================

def load_binary_mask(mask_path):

    mask = tf.io.decode_image(
        tf.io.read_file(str(mask_path)),
        channels=1,
        expand_animations=False,
    )

    mask.set_shape([None, None, 1])

    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    mask = tf.cast(mask, tf.float32)

    mask = tf.cast(
        mask >= 128.0,
        tf.uint8,
    )

    return mask.numpy().squeeze()


# ============================================================
# 3. LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = tf.io.decode_image(
        tf.io.read_file(str(image_path)),
        channels=1,
        expand_animations=False,
    )

    image.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image.numpy()


# ============================================================
# 4. CALCULATE PREDICTION ERRORS
# ============================================================

def calculate_errors(
    ground_truth,
    predicted_mask,
    lung_mask,
):

    ground_truth = ground_truth.astype(bool)
    predicted_mask = predicted_mask.astype(bool)
    lung_mask = lung_mask.astype(bool)

    tp = int(
        np.logical_and(
            ground_truth,
            predicted_mask,
        ).sum()
    )

    fp = int(
        np.logical_and(
            np.logical_not(ground_truth),
            predicted_mask,
        ).sum()
    )

    fn = int(
        np.logical_and(
            ground_truth,
            np.logical_not(predicted_mask),
        ).sum()
    )

    tn = int(
        np.logical_and(
            np.logical_not(ground_truth),
            np.logical_not(predicted_mask),
        ).sum()
    )

    gt_pixels = int(ground_truth.sum())
    pred_pixels = int(predicted_mask.sum())
    lung_pixels = int(lung_mask.sum())

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

    if gt_pixels > 0:
        area_ratio = pred_pixels / gt_pixels
    else:
        area_ratio = np.nan

    if lung_pixels > 0:
        gt_lung_ratio = (
            gt_pixels / lung_pixels
        ) * 100.0

        pred_lung_ratio = (
            pred_pixels / lung_pixels
        ) * 100.0
    else:
        gt_lung_ratio = np.nan
        pred_lung_ratio = np.nan

    return {

        "ground_truth_pixels": gt_pixels,

        "predicted_pixels": pred_pixels,

        "true_positive_pixels": tp,

        "false_positive_pixels": fp,

        "false_negative_pixels": fn,

        "true_negative_pixels": tn,

        "dice": float(dice),

        "iou": float(iou),

        "precision": float(precision),

        "recall": float(recall),

        "predicted_to_reference_area_ratio": float(
            area_ratio
        ),

        "reference_infection_lung_percent": float(
            gt_lung_ratio
        ),

        "predicted_infection_lung_percent": float(
            pred_lung_ratio
        ),

        "false_positive_percent_of_prediction": float(
            fp / (pred_pixels + EPSILON) * 100.0
        ),

        "false_negative_percent_of_reference": float(
            fn / (gt_pixels + EPSILON) * 100.0
        ),
    }


# ============================================================
# 5. SUBGROUP SUMMARY
# ============================================================

def create_subgroup_summary(data):

    records = []

    for group, subset in data.groupby(
        "annotation_group",
        sort=False,
    ):

        records.append({

            "annotation_group": group,

            "sample_count": int(len(subset)),

            "mean_dice": float(
                subset["dice"].mean()
            ),

            "mean_iou": float(
                subset["iou"].mean()
            ),

            "mean_precision": float(
                subset["precision"].mean()
            ),

            "mean_recall": float(
                subset["recall"].mean()
            ),

            "mean_ground_truth_pixels": float(
                subset["ground_truth_pixels"].mean()
            ),

            "mean_predicted_pixels": float(
                subset["predicted_pixels"].mean()
            ),

            "mean_false_positive_pixels": float(
                subset["false_positive_pixels"].mean()
            ),

            "mean_false_negative_pixels": float(
                subset["false_negative_pixels"].mean()
            ),

            "mean_predicted_to_reference_area_ratio": float(
                subset[
                    "predicted_to_reference_area_ratio"
                ].mean()
            ),

            "median_predicted_to_reference_area_ratio": float(
                subset[
                    "predicted_to_reference_area_ratio"
                ].median()
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COVID INFECTION SEGMENTATION - PREDICTION ERROR ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        MODEL_PATH,
        SUBGROUP_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("\nLoading subgroup information...")

    subgroup_data = pd.read_csv(
        SUBGROUP_PATH
    )

    print(
        "Test samples:",
        len(subgroup_data),
    )

    print("\nLoading saved infection U-Net...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    records = []

    print("\nCalculating pixel-level prediction errors...")

    for index, row in subgroup_data.iterrows():

        image_name = str(row["image"])

        image_path = IMAGE_DIR / f"{image_name}.png"

        infection_path = (
            INFECTION_MASK_DIR / f"{image_name}.png"
        )

        lung_path = LUNG_MASK_DIR / f"{image_name}.png"

        for path in [
            image_path,
            infection_path,
            lung_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(path)

        image = load_image(image_path)

        ground_truth = load_binary_mask(
            infection_path
        )

        lung_mask = load_binary_mask(
            lung_path
        )

        prediction = model(
            image[np.newaxis, ...],
            training=False,
        ).numpy()[0, :, :, 0]

        predicted_mask = (
            prediction >= THRESHOLD
        ).astype(np.uint8)

        errors = calculate_errors(
            ground_truth,
            predicted_mask,
            lung_mask,
        )

        record = {
            "image": image_name,
            "annotation_group": row["annotation_group"],
            **errors,
        }

        records.append(record)

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(subgroup_data)} images"
            )

    # --------------------------------------------------------
    # PER-IMAGE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(records)

    per_image_path = (
        OUTPUT_DIR
        / "test_prediction_error_per_image.csv"
    )

    results.to_csv(
        per_image_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUBGROUP RESULTS
    # --------------------------------------------------------

    subgroup_summary = create_subgroup_summary(
        results
    )

    subgroup_path = (
        OUTPUT_DIR
        / "test_prediction_error_by_subgroup.csv"
    )

    subgroup_summary.to_csv(
        subgroup_path,
        index=False,
    )

    # --------------------------------------------------------
    # GLOBAL PIXEL COUNTS
    # --------------------------------------------------------

    total_tp = int(
        results["true_positive_pixels"].sum()
    )

    total_fp = int(
        results["false_positive_pixels"].sum()
    )

    total_fn = int(
        results["false_negative_pixels"].sum()
    )

    total_tn = int(
        results["true_negative_pixels"].sum()
    )

    total_gt = int(
        results["ground_truth_pixels"].sum()
    )

    total_pred = int(
        results["predicted_pixels"].sum()
    )

    global_summary = {

        "test_images": int(len(results)),

        "prediction_threshold": THRESHOLD,

        "global_true_positive_pixels": total_tp,

        "global_false_positive_pixels": total_fp,

        "global_false_negative_pixels": total_fn,

        "global_true_negative_pixels": total_tn,

        "global_ground_truth_pixels": total_gt,

        "global_predicted_pixels": total_pred,

        "global_dice": float(
            2.0 * total_tp
            / (2.0 * total_tp + total_fp + total_fn + EPSILON)
        ),

        "global_iou": float(
            total_tp
            / (total_tp + total_fp + total_fn + EPSILON)
        ),

        "global_precision": float(
            total_tp
            / (total_tp + total_fp + EPSILON)
        ),

        "global_recall": float(
            total_tp
            / (total_tp + total_fn + EPSILON)
        ),

        "note": (
            "Pixel-level errors are measured against supplied "
            "COVID-QU-Ex infection annotations. They do not "
            "establish clinical correctness."
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "test_prediction_error_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            global_summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GLOBAL PIXEL-LEVEL ERROR SUMMARY")
    print("=" * 70)

    print("True positive pixels:", total_tp)
    print("False positive pixels:", total_fp)
    print("False negative pixels:", total_fn)
    print("True negative pixels:", total_tn)

    print("\nGlobal Dice:", round(global_summary["global_dice"], 4))
    print("Global IoU:", round(global_summary["global_iou"], 4))
    print(
        "Global Precision:",
        round(global_summary["global_precision"], 4),
    )
    print(
        "Global Recall:",
        round(global_summary["global_recall"], 4),
    )

    print("\nSUBGROUP ERROR ANALYSIS")
    print("-" * 110)

    print(
        subgroup_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved files:")
    print(per_image_path)
    print(subgroup_path)
    print(json_path)

    print("\nPREDICTION ERROR ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/97_analyze_infection_prediction_errors.py