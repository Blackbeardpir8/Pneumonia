
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val"
    / "COVID-19"
)

IMAGE_DIR = DATA_DIR / "images"
INFECTION_MASK_DIR = DATA_DIR / "infection masks"
LUNG_MASK_DIR = DATA_DIR / "lung masks"

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "lung_constraint_analysis"
)

IMAGE_SIZE = 256
MASK_THRESHOLD = 128
PREDICTION_THRESHOLD = 0.40

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE HELPERS
# ============================================================

def find_matching_file(folder, stem):

    matches = list(folder.glob(stem + ".*"))

    if not matches:
        raise FileNotFoundError(
            f"Matching file not found: {stem} in {folder}"
        )

    return matches[0]


# ============================================================
# IMAGE AND MASK LOADING
# ============================================================

def load_image(path):

    image = Image.open(path).convert("L")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return image / 255.0


def load_mask(path):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(mask)

    return (
        mask >= MASK_THRESHOLD
    ).astype(np.uint8)


# ============================================================
# ANNOTATION GROUP
# ============================================================

def get_annotation_group(infection_mask, lung_mask):

    if np.array_equal(infection_mask, lung_mask):
        return "IDENTICAL_TO_LUNG"

    lung_area = lung_mask.sum()

    if lung_area == 0:
        return "INVALID"

    ratio = infection_mask.sum() / lung_area

    if ratio < 0.10:
        return "NON_IDENTICAL_SMALL"

    elif ratio < 0.50:
        return "NON_IDENTICAL_MODERATE"

    return "NON_IDENTICAL_LARGE"


# ============================================================
# METRICS
# ============================================================

def calculate_pixel_metrics(reference, prediction):

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    tp = int(
        np.logical_and(reference, prediction).sum()
    )

    fp = int(
        np.logical_and(~reference, prediction).sum()
    )

    fn = int(
        np.logical_and(reference, ~prediction).sum()
    )

    tn = int(
        np.logical_and(~reference, ~prediction).sum()
    )

    denominator_dice = (2 * tp) + fp + fn
    denominator_iou = tp + fp + fn

    dice = (
        (2 * tp) / denominator_dice
        if denominator_dice > 0 else 1.0
    )

    iou = (
        tp / denominator_iou
        if denominator_iou > 0 else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 1.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 1.0
    )

    reference_area = int(reference.sum())
    predicted_area = int(prediction.sum())

    area_ratio = (
        predicted_area / reference_area
        if reference_area > 0 else np.nan
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "reference_area": reference_area,
        "predicted_area": predicted_area,
        "predicted_reference_area_ratio": area_ratio
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_probability_map(model, image):

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    probability_map = model.predict(
        model_input,
        verbose=0
    )

    return probability_map[0, :, :, 0]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("LUNG-CONSTRAINED VS UNCONSTRAINED INFECTION SEGMENTATION")
    print("=" * 75)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nInfection U-Net loaded successfully.")

    image_files = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
    ])

    print(f"Validation images found: {len(image_files)}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")

    per_image_results = []

    # Store global pixel counts separately for each method.
    global_counts = {
        "unconstrained": {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0
        },
        "lung_constrained": {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0
        }
    }

    outside_lung_prediction_pixels = 0
    original_prediction_pixels = 0
    reference_pixels_outside_lung = 0

    for index, image_path in enumerate(image_files, start=1):

        stem = image_path.stem

        infection_path = find_matching_file(
            INFECTION_MASK_DIR,
            stem
        )

        lung_path = find_matching_file(
            LUNG_MASK_DIR,
            stem
        )

        image = load_image(image_path)

        reference = load_mask(infection_path)

        lung_mask = load_mask(lung_path)

        group = get_annotation_group(
            reference,
            lung_mask
        )

        if group == "INVALID":
            print(f"Skipping invalid lung mask: {stem}")
            continue

        probability_map = predict_probability_map(
            model,
            image
        )

        # Original prediction without lung restriction.
        unconstrained_prediction = (
            probability_map >= PREDICTION_THRESHOLD
        ).astype(np.uint8)

        # Apply lung mask as a spatial restriction.
        lung_constrained_prediction = (
            unconstrained_prediction * lung_mask
        ).astype(np.uint8)

        unconstrained_metrics = calculate_pixel_metrics(
            reference,
            unconstrained_prediction
        )

        constrained_metrics = calculate_pixel_metrics(
            reference,
            lung_constrained_prediction
        )

        for key in ["tp", "fp", "fn", "tn"]:

            global_counts["unconstrained"][key] += (
                unconstrained_metrics[key]
            )

            global_counts["lung_constrained"][key] += (
                constrained_metrics[key]
            )

        # Prediction pixels outside the lung mask.
        outside_pixels = np.logical_and(
            unconstrained_prediction == 1,
            lung_mask == 0
        ).sum()

        outside_lung_prediction_pixels += int(
            outside_pixels
        )

        original_prediction_pixels += int(
            unconstrained_prediction.sum()
        )

        reference_pixels_outside_lung += int(
            np.logical_and(
                reference == 1,
                lung_mask == 0
            ).sum()
        )

        per_image_results.append({

            "image_name": image_path.name,
            "annotation_group": group,

            "reference_area": reference.sum(),
            "lung_area": lung_mask.sum(),

            "unconstrained_dice":
                unconstrained_metrics["dice"],

            "constrained_dice":
                constrained_metrics["dice"],

            "unconstrained_iou":
                unconstrained_metrics["iou"],

            "constrained_iou":
                constrained_metrics["iou"],

            "unconstrained_precision":
                unconstrained_metrics["precision"],

            "constrained_precision":
                constrained_metrics["precision"],

            "unconstrained_recall":
                unconstrained_metrics["recall"],

            "constrained_recall":
                constrained_metrics["recall"],

            "unconstrained_fp":
                unconstrained_metrics["fp"],

            "constrained_fp":
                constrained_metrics["fp"],

            "unconstrained_fn":
                unconstrained_metrics["fn"],

            "constrained_fn":
                constrained_metrics["fn"],

            "prediction_pixels_outside_lung":
                int(outside_pixels),

            "unconstrained_area_ratio":
                unconstrained_metrics[
                    "predicted_reference_area_ratio"
                ],

            "constrained_area_ratio":
                constrained_metrics[
                    "predicted_reference_area_ratio"
                ],

            "dice_improvement":
                constrained_metrics["dice"]
                - unconstrained_metrics["dice"]

        })

        if index % 100 == 0 or index == len(image_files):

            print(
                f"Processed {index}/{len(image_files)} images"
            )

    # ========================================================
    # SAVE PER-IMAGE RESULTS
    # ========================================================

    results_df = pd.DataFrame(per_image_results)

    per_image_path = (
        OUTPUT_DIR
        / "lung_constraint_per_image.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # ========================================================
    # GLOBAL PIXEL METRICS
    # ========================================================

    global_rows = []

    for method, counts in global_counts.items():

        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]

        dice_denominator = (2 * tp) + fp + fn
        iou_denominator = tp + fp + fn

        dice = (
            (2 * tp) / dice_denominator
            if dice_denominator > 0 else 1.0
        )

        iou = (
            tp / iou_denominator
            if iou_denominator > 0 else 1.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0 else 1.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0 else 1.0
        )

        global_rows.append({

            "method": method,

            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,

            "global_dice": dice,
            "global_iou": iou,
            "global_precision": precision,
            "global_recall": recall

        })

    global_df = pd.DataFrame(global_rows)

    global_path = (
        OUTPUT_DIR
        / "lung_constraint_global_summary.csv"
    )

    global_df.to_csv(
        global_path,
        index=False
    )

    # ========================================================
    # MEAN PER-IMAGE COMPARISON
    # ========================================================

    metric_columns = [
        "unconstrained_dice",
        "constrained_dice",
        "unconstrained_iou",
        "constrained_iou",
        "unconstrained_precision",
        "constrained_precision",
        "unconstrained_recall",
        "constrained_recall",
        "unconstrained_fp",
        "constrained_fp",
        "unconstrained_fn",
        "constrained_fn",
        "dice_improvement"
    ]

    mean_summary = results_df[metric_columns].mean()

    print("\n" + "=" * 75)
    print("MEAN PER-IMAGE COMPARISON")
    print("=" * 75)

    print(mean_summary.to_string())

    # ========================================================
    # SUBGROUP ANALYSIS
    # ========================================================

    subgroup_rows = []

    for group, group_df in results_df.groupby(
        "annotation_group"
    ):

        subgroup_rows.append({

            "annotation_group": group,

            "sample_count": len(group_df),

            "unconstrained_mean_dice":
                group_df["unconstrained_dice"].mean(),

            "constrained_mean_dice":
                group_df["constrained_dice"].mean(),

            "unconstrained_mean_iou":
                group_df["unconstrained_iou"].mean(),

            "constrained_mean_iou":
                group_df["constrained_iou"].mean(),

            "unconstrained_mean_precision":
                group_df["unconstrained_precision"].mean(),

            "constrained_mean_precision":
                group_df["constrained_precision"].mean(),

            "unconstrained_mean_recall":
                group_df["unconstrained_recall"].mean(),

            "constrained_mean_recall":
                group_df["constrained_recall"].mean(),

            "mean_dice_improvement":
                group_df["dice_improvement"].mean(),

            "mean_unconstrained_fp":
                group_df["unconstrained_fp"].mean(),

            "mean_constrained_fp":
                group_df["constrained_fp"].mean()

        })

    subgroup_df = pd.DataFrame(subgroup_rows)

    subgroup_path = (
        OUTPUT_DIR
        / "lung_constraint_subgroup_summary.csv"
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False
    )

    # ========================================================
    # OUTSIDE-LUNG ANALYSIS
    # ========================================================

    outside_percentage = (
        100.0 * outside_lung_prediction_pixels
        / original_prediction_pixels
        if original_prediction_pixels > 0 else 0.0
    )

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {

        "validation_sample_count": len(results_df),

        "prediction_threshold": PREDICTION_THRESHOLD,

        "outside_lung_prediction_pixels":
            outside_lung_prediction_pixels,

        "original_prediction_pixels":
            original_prediction_pixels,

        "percentage_prediction_pixels_outside_lung":
            outside_percentage,

        "reference_pixels_outside_lung":
            reference_pixels_outside_lung,

        "mean_unconstrained_dice":
            float(results_df["unconstrained_dice"].mean()),

        "mean_constrained_dice":
            float(results_df["constrained_dice"].mean()),

        "mean_dice_improvement":
            float(results_df["dice_improvement"].mean()),

        "global_pixel_metrics":
            global_rows,

        "interpretation_note":
            "Lung constraint is a post-processing ablation. "
            "Results do not establish clinical validity or "
            "prove that all predictions outside the reference "
            "lung mask are clinically incorrect."

    }

    json_path = (
        OUTPUT_DIR
        / "lung_constraint_analysis_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 75)
    print("GLOBAL PIXEL METRICS")
    print("=" * 75)

    print(
        global_df.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("SUBGROUP COMPARISON")
    print("=" * 75)

    print(
        subgroup_df.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("OUTSIDE-LUNG PREDICTION ANALYSIS")
    print("=" * 75)

    print(
        f"Prediction pixels outside lung: "
        f"{outside_lung_prediction_pixels:,}"
    )

    print(
        f"Total original prediction pixels: "
        f"{original_prediction_pixels:,}"
    )

    print(
        f"Percentage outside lung: "
        f"{outside_percentage:.2f}%"
    )

    print(
        f"Reference infection pixels outside lung: "
        f"{reference_pixels_outside_lung:,}"
    )

    print("\nSaved files:")
    print(per_image_path)
    print(global_path)
    print(subgroup_path)
    print(json_path)

    print("\nLUNG CONSTRAINT ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
