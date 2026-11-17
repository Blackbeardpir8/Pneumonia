
"""
SCRIPT 108
Threshold Sensitivity by Infection Area Group

Evaluates the COVID infection U-Net at multiple thresholds.

Compares:
- Unconstrained predictions
- Automatically predicted-lung-constrained predictions

Evaluates:
- Overall mean per-image Dice
- Global pixel Dice
- Precision and recall
- Performance by reference infection-area group

No model training is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    ROOT / "data" / "raw" / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val" / "COVID-19" / "images"
)

REFERENCE_DIR = (
    ROOT / "data" / "raw" / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val" / "COVID-19" / "infection masks"
)

LUNG_MASK_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
    / "predicted_lung_masks"
)

MODEL_PATH = (
    ROOT / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

OUTPUT_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "threshold_sensitivity_by_area_group"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 8

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


# ============================================================
# FILE LOOKUP
# ============================================================

VALID_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def build_lookup(directory, remove_suffix=None):

    if not directory.exists():
        raise FileNotFoundError(directory)

    lookup = {}

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        stem = path.stem.lower()

        if remove_suffix and stem.endswith(remove_suffix):
            stem = stem[:-len(remove_suffix)]

        lookup[stem] = path

    return lookup


def get_file(image_name, lookup, description):

    stem = Path(image_name).stem.lower()

    if stem not in lookup:
        raise FileNotFoundError(
            f"{description} missing for {image_name}"
        )

    return lookup[stem]


# ============================================================
# PREPROCESSING
# ============================================================

def load_grayscale(path, size=IMAGE_SIZE):

    image = Image.open(path).convert("L")

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0


def load_binary_mask(path, size=IMAGE_SIZE):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (size, size),
        Image.Resampling.NEAREST,
    )

    return np.asarray(mask) >= 128


# ============================================================
# METRICS
# ============================================================

def calculate_counts(reference, prediction):

    tp = int(np.logical_and(reference, prediction).sum())
    fp = int(np.logical_and(~reference, prediction).sum())
    fn = int(np.logical_and(reference, ~prediction).sum())
    tn = int(np.logical_and(~reference, ~prediction).sum())

    return tp, fp, fn, tn


def calculate_metrics(reference, prediction):

    tp, fp, fn, tn = calculate_counts(
        reference,
        prediction,
    )

    dice_denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator
        else 1.0
    )

    iou_denominator = tp + fp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 1.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def assign_area_group(area_percentage):

    if area_percentage < 1:
        return "VERY_SMALL"

    if area_percentage < 10:
        return "SMALL"

    if area_percentage < 50:
        return "MODERATE"

    return "LARGE"


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def summarize_results(df):

    summary_rows = []

    for (threshold, strategy), group in df.groupby(
        ["threshold", "strategy"],
        observed=True,
    ):

        total_tp = int(group["tp"].sum())
        total_fp = int(group["fp"].sum())
        total_fn = int(group["fn"].sum())

        global_dice = (
            2 * total_tp
            / (2 * total_tp + total_fp + total_fn)
        )

        global_iou = (
            total_tp
            / (total_tp + total_fp + total_fn)
        )

        global_precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp else 1.0
        )

        global_recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn else 1.0
        )

        summary_rows.append({
            "threshold": float(threshold),
            "strategy": strategy,
            "image_count": int(len(group)),
            "mean_image_dice": float(group["dice"].mean()),
            "median_image_dice": float(group["dice"].median()),
            "global_pixel_dice": float(global_dice),
            "global_pixel_iou": float(global_iou),
            "global_precision": float(global_precision),
            "global_recall": float(global_recall),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        })

    return pd.DataFrame(summary_rows)


def summarize_area_groups(df):

    rows = []

    for (
        threshold,
        strategy,
        area_group,
    ), group in df.groupby(
        ["threshold", "strategy", "area_group"],
        observed=True,
    ):

        rows.append({
            "threshold": float(threshold),
            "strategy": strategy,
            "area_group": area_group,
            "image_count": int(len(group)),
            "mean_reference_area_percentage": float(
                group["reference_area_percentage"].mean()
            ),
            "mean_dice": float(group["dice"].mean()),
            "median_dice": float(group["dice"].median()),
            "mean_precision": float(group["precision"].mean()),
            "mean_recall": float(group["recall"].mean()),
            "mean_predicted_reference_area_ratio": float(
                group["predicted_reference_area_ratio"].mean()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("THRESHOLD SENSITIVITY BY INFECTION AREA")
    print("=" * 80)

    for path in [IMAGE_DIR, REFERENCE_DIR, LUNG_MASK_DIR, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    image_lookup = build_lookup(IMAGE_DIR)
    reference_lookup = build_lookup(REFERENCE_DIR)

    lung_lookup = build_lookup(
        LUNG_MASK_DIR,
        remove_suffix="_lung_prediction",
    )

    print(f"\nImages: {len(image_lookup)}")
    print(f"Reference masks: {len(reference_lookup)}")
    print(f"Predicted lung masks: {len(lung_lookup)}")

    image_names = sorted(
        set(image_lookup)
        & set(reference_lookup)
        & set(lung_lookup)
    )

    if len(image_names) != 466:
        raise ValueError(
            f"Expected 466 matched validation images, "
            f"found {len(image_names)}. "
            "Do not continue until file matching is checked."
        )

    print(f"Matched validation images: {len(image_names)}")

    print("\nLoading infection U-Net...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Infection U-Net loaded.")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # --------------------------------------------------------
    # Load all validation inputs.
    # --------------------------------------------------------

    images = []
    references = []
    lung_masks = []
    metadata = []

    for index, stem in enumerate(image_names):

        image_path = image_lookup[stem]
        reference_path = reference_lookup[stem]
        lung_path = lung_lookup[stem]

        image = load_grayscale(image_path)

        reference = load_binary_mask(reference_path)
        lung = load_binary_mask(lung_path)

        reference_area = int(reference.sum())
        total_pixels = int(reference.size)

        area_percentage = (
            reference_area / total_pixels * 100
        )

        images.append(image[..., np.newaxis])
        references.append(reference)
        lung_masks.append(lung)

        metadata.append({
            "image_name": image_path.name,
            "reference_area": reference_area,
            "reference_area_percentage": area_percentage,
            "area_group": assign_area_group(area_percentage),
        })

        if (index + 1) % 100 == 0:
            print(
                f"Loaded {index + 1}/{len(image_names)} images"
            )

    images = np.asarray(images, dtype=np.float32)
    references = np.asarray(references, dtype=bool)
    lung_masks = np.asarray(lung_masks, dtype=bool)

    # --------------------------------------------------------
    # Predict once per image, then evaluate all thresholds.
    # --------------------------------------------------------

    print("\nGenerating infection probability maps...")

    probabilities = []

    for start in range(0, len(images), BATCH_SIZE):

        batch = images[start:start + BATCH_SIZE]

        prediction = model.predict(
            batch,
            verbose=0,
        )

        probabilities.append(
            np.asarray(prediction, dtype=np.float32)
        )

        print(
            f"Predicted {min(start + BATCH_SIZE, len(images))}"
            f"/{len(images)} images"
        )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    if probabilities.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected prediction shape: {probabilities.shape}"
        )

    # Remove the single output channel.
    probabilities = probabilities[..., 0]

    print("\nProbability maps generated.")

    # --------------------------------------------------------
    # Evaluate thresholds.
    # --------------------------------------------------------

    result_rows = []

    for threshold in THRESHOLDS:

        print(f"\nEvaluating threshold: {threshold:.2f}")

        for index, meta in enumerate(metadata):

            reference = references[index]
            lung = lung_masks[index]

            unconstrained_prediction = (
                probabilities[index] >= threshold
            )

            constrained_prediction = (
                unconstrained_prediction & lung
            )

            for strategy, prediction in [
                (
                    "UNCONSTRAINED",
                    unconstrained_prediction,
                ),
                (
                    "AUTOMATIC_LUNG_CONSTRAINED",
                    constrained_prediction,
                ),
            ]:

                metrics = calculate_metrics(
                    reference,
                    prediction,
                )

                predicted_area = int(prediction.sum())

                area_ratio = (
                    predicted_area / meta["reference_area"]
                    if meta["reference_area"] > 0
                    else None
                )

                result_rows.append({
                    "image_name": meta["image_name"],
                    "threshold": threshold,
                    "strategy": strategy,
                    "area_group": meta["area_group"],
                    "reference_area_percentage":
                        meta["reference_area_percentage"],
                    "predicted_area": predicted_area,
                    "predicted_reference_area_ratio": area_ratio,
                    **metrics,
                })

    results = pd.DataFrame(result_rows)

    results.to_csv(
        OUTPUT_DIR / "threshold_per_image_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Overall comparison.
    # --------------------------------------------------------

    overall = summarize_results(results)

    overall.to_csv(
        OUTPUT_DIR / "threshold_overall_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Area group comparison.
    # --------------------------------------------------------

    area_summary = summarize_area_groups(results)

    area_summary.to_csv(
        OUTPUT_DIR / "threshold_area_group_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Find best threshold by group and strategy.
    # --------------------------------------------------------

    best_rows = []

    for (strategy, area_group), group in area_summary.groupby(
        ["strategy", "area_group"],
        observed=True,
    ):

        best = group.loc[group["mean_dice"].idxmax()]

        best_rows.append({
            "strategy": strategy,
            "area_group": area_group,
            "best_threshold_by_mean_dice":
                float(best["threshold"]),
            "best_mean_dice": float(best["mean_dice"]),
            "image_count": int(best["image_count"]),
        })

    best_df = pd.DataFrame(best_rows)

    best_df.to_csv(
        OUTPUT_DIR / "best_threshold_by_area_group.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("OVERALL THRESHOLD COMPARISON")
    print("=" * 80)

    print(overall.to_string(index=False))

    print("\n" + "=" * 80)
    print("AREA GROUP THRESHOLD COMPARISON")
    print("=" * 80)

    print(area_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("BEST THRESHOLD BY AREA GROUP")
    print("=" * 80)

    print(best_df.to_string(index=False))

    # --------------------------------------------------------
    # Save summary JSON.
    # --------------------------------------------------------

    summary = {
        "study": "Threshold sensitivity by infection area",
        "validation_images": int(len(image_names)),
        "thresholds": THRESHOLDS,
        "model_path": str(MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "preprocessing": "Grayscale, resize 256x256, normalize 0-1",
        "best_threshold_by_area_group":
            best_rows,
        "overall_results":
            overall.to_dict(orient="records"),
        "limitations": [
            "Thresholds are evaluated on the validation set.",
            "Selecting thresholds on validation data can overfit.",
            "Reference infection masks are evaluation targets.",
            "Area groups describe annotation geometry, not clinical severity.",
            "Results apply to this COVID infection segmentation dataset.",
            "No clinical diagnostic claim is established.",
        ],
    }

    with open(
        OUTPUT_DIR / "threshold_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 80)
    print("SCRIPT 108 COMPLETED")
    print("=" * 80)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. threshold_per_image_results.csv")
    print("2. threshold_overall_comparison.csv")
    print("3. threshold_area_group_comparison.csv")
    print("4. best_threshold_by_area_group.csv")
    print("5. threshold_sensitivity_summary.json")


if __name__ == "__main__":
    main()

# # to run this script, use the following command:
# python scripts/108_threshold_sensitivity_by_infection_area.py