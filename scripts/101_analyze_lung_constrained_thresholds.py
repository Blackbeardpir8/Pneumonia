
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
    / "constrained_threshold_analysis"
)

IMAGE_SIZE = 256
MASK_THRESHOLD = 128

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

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

    image = np.asarray(image, dtype=np.float32)

    return image / 255.0


def load_mask(path):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(mask)

    return (mask >= MASK_THRESHOLD).astype(np.uint8)


# ============================================================
# ANNOTATION GROUP
# ============================================================

def determine_annotation_group(reference, lung_mask):

    if np.array_equal(reference, lung_mask):
        return "IDENTICAL_TO_LUNG"

    lung_area = lung_mask.sum()

    if lung_area == 0:
        return "INVALID"

    ratio = reference.sum() / lung_area

    if ratio < 0.10:
        return "NON_IDENTICAL_SMALL"

    elif ratio < 0.50:
        return "NON_IDENTICAL_MODERATE"

    return "NON_IDENTICAL_LARGE"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(reference, prediction):

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

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_probability_map(model, image):

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )

    return prediction[0, :, :, 0]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("LUNG-CONSTRAINED THRESHOLD OPTIMIZATION")
    print("=" * 75)

    print("\nThresholds:", THRESHOLDS)

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

    print(f"Validation images: {len(image_files)}")

    per_image_rows = []

    # Global pixel counts for every threshold and method.
    global_counts = {}

    for threshold in THRESHOLDS:

        global_counts[threshold] = {
            "unconstrained": {
                "tp": 0, "fp": 0, "fn": 0, "tn": 0
            },
            "lung_constrained": {
                "tp": 0, "fp": 0, "fn": 0, "tn": 0
            }
        }

    # ========================================================
    # PROCESS VALIDATION IMAGES
    # ========================================================

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

        group = determine_annotation_group(
            reference,
            lung_mask
        )

        if group == "INVALID":
            print(f"Skipping invalid sample: {stem}")
            continue

        # One model prediction per image.
        probability_map = predict_probability_map(
            model,
            image
        )

        for threshold in THRESHOLDS:

            # Original prediction.
            unconstrained = (
                probability_map >= threshold
            ).astype(np.uint8)

            # Restrict prediction to lung region.
            constrained = (
                unconstrained * lung_mask
            ).astype(np.uint8)

            methods = {
                "unconstrained": unconstrained,
                "lung_constrained": constrained
            }

            row = {
                "image_name": image_path.name,
                "annotation_group": group,
                "threshold": threshold
            }

            for method_name, prediction in methods.items():

                metrics = calculate_metrics(
                    reference,
                    prediction
                )

                # Accumulate global pixel counts.
                for key in ["tp", "fp", "fn", "tn"]:

                    global_counts[threshold][method_name][key] += (
                        metrics[key]
                    )

                row.update({

                    f"{method_name}_dice":
                        metrics["dice"],

                    f"{method_name}_iou":
                        metrics["iou"],

                    f"{method_name}_precision":
                        metrics["precision"],

                    f"{method_name}_recall":
                        metrics["recall"],

                    f"{method_name}_fp":
                        metrics["fp"],

                    f"{method_name}_fn":
                        metrics["fn"],

                    f"{method_name}_predicted_area":
                        int(prediction.sum())

                })

            row["dice_improvement"] = (
                row["lung_constrained_dice"]
                - row["unconstrained_dice"]
            )

            per_image_rows.append(row)

        if index % 100 == 0 or index == len(image_files):

            print(
                f"Processed {index}/{len(image_files)} images"
            )

    # ========================================================
    # PER-IMAGE DATAFRAME
    # ========================================================

    per_image_df = pd.DataFrame(per_image_rows)

    per_image_path = (
        OUTPUT_DIR
        / "constrained_threshold_per_image.csv"
    )

    per_image_df.to_csv(
        per_image_path,
        index=False
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    global_rows = []

    for threshold in THRESHOLDS:

        for method_name in [
            "unconstrained",
            "lung_constrained"
        ]:

            counts = global_counts[threshold][method_name]

            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            tn = counts["tn"]

            dice_denominator = (2 * tp) + fp + fn
            iou_denominator = tp + fp + fn

            global_dice = (
                (2 * tp) / dice_denominator
                if dice_denominator > 0 else 1.0
            )

            global_iou = (
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

                "threshold": threshold,
                "method": method_name,

                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,

                "global_dice": global_dice,
                "global_iou": global_iou,
                "global_precision": precision,
                "global_recall": recall

            })

    global_df = pd.DataFrame(global_rows)

    global_path = (
        OUTPUT_DIR
        / "constrained_threshold_global_summary.csv"
    )

    global_df.to_csv(
        global_path,
        index=False
    )

    # ========================================================
    # MEAN PER-IMAGE SUMMARY
    # ========================================================

    metric_columns = [
        "unconstrained_dice",
        "lung_constrained_dice",
        "unconstrained_iou",
        "lung_constrained_iou",
        "unconstrained_precision",
        "lung_constrained_precision",
        "unconstrained_recall",
        "lung_constrained_recall",
        "unconstrained_fp",
        "lung_constrained_fp",
        "unconstrained_fn",
        "lung_constrained_fn",
        "dice_improvement"
    ]

    mean_df = (
        per_image_df
        .groupby("threshold")[metric_columns]
        .mean()
        .reset_index()
    )

    mean_path = (
        OUTPUT_DIR
        / "constrained_threshold_mean_summary.csv"
    )

    mean_df.to_csv(
        mean_path,
        index=False
    )

    # ========================================================
    # SUBGROUP SUMMARY
    # ========================================================

    subgroup_metrics = [
        "unconstrained_dice",
        "lung_constrained_dice",
        "unconstrained_iou",
        "lung_constrained_iou",
        "unconstrained_precision",
        "lung_constrained_precision",
        "unconstrained_recall",
        "lung_constrained_recall",
        "dice_improvement",
        "unconstrained_fp",
        "lung_constrained_fp"
    ]

    subgroup_df = (
        per_image_df
        .groupby(
            ["threshold", "annotation_group"]
        )[subgroup_metrics]
        .mean()
        .reset_index()
    )

    subgroup_counts = (
        per_image_df
        .groupby(
            ["threshold", "annotation_group"]
        )["image_name"]
        .nunique()
        .reset_index(name="sample_count")
    )

    subgroup_df = subgroup_df.merge(
        subgroup_counts,
        on=["threshold", "annotation_group"]
    )

    subgroup_path = (
        OUTPUT_DIR
        / "constrained_threshold_subgroup_summary.csv"
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False
    )

    # ========================================================
    # SELECT BEST VALIDATION THRESHOLDS
    # ========================================================

    best_mean_index = mean_df[
        "lung_constrained_dice"
    ].idxmax()

    best_mean_row = mean_df.loc[best_mean_index]

    best_global_index = global_df[
        global_df["method"] == "lung_constrained"
    ]["global_dice"].idxmax()

    best_global_row = global_df.loc[best_global_index]

    # ========================================================
    # JSON SUMMARY
    # ========================================================

    summary = {

        "validation_sample_count":
            int(per_image_df["image_name"].nunique()),

        "thresholds_evaluated":
            THRESHOLDS,

        "best_threshold_by_mean_per_image_dice":
            float(best_mean_row["threshold"]),

        "best_mean_per_image_dice":
            float(best_mean_row["lung_constrained_dice"]),

        "best_threshold_by_global_pixel_dice":
            float(best_global_row["threshold"]),

        "best_global_pixel_dice":
            float(best_global_row["global_dice"]),

        "selection_dataset":
            "COVID-QU-Ex validation subset",

        "selection_note":
            "Threshold selection is based on validation data. "
            "The held-out test set must not be used for threshold tuning.",

        "interpretation_note":
            "Lung-constrained predictions are obtained by "
            "multiplying binary infection predictions by the "
            "reference lung mask. Results are an experimental "
            "post-processing comparison and do not establish "
            "clinical validity."

    }

    json_path = (
        OUTPUT_DIR
        / "constrained_threshold_analysis_summary.json"
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
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("MEAN PER-IMAGE THRESHOLD RESULTS")
    print("=" * 75)

    print(
        mean_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("GLOBAL PIXEL THRESHOLD RESULTS")
    print("=" * 75)

    print(
        global_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("SUBGROUP THRESHOLD RESULTS")
    print("=" * 75)

    print(
        subgroup_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("BEST VALIDATION THRESHOLDS")
    print("=" * 75)

    print(
        f"Best threshold by mean per-image Dice: "
        f"{best_mean_row['threshold']:.2f}"
    )

    print(
        f"Best mean per-image Dice: "
        f"{best_mean_row['lung_constrained_dice']:.4f}"
    )

    print(
        f"Best threshold by global pixel Dice: "
        f"{best_global_row['threshold']:.2f}"
    )

    print(
        f"Best global pixel Dice: "
        f"{best_global_row['global_dice']:.4f}"
    )

    print("\nSaved files:")
    print(per_image_path)
    print(mean_path)
    print(global_path)
    print(subgroup_path)
    print(json_path)

    print("\nCONSTRAINED THRESHOLD ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()



