
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

LUNG_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

INFECTION_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
)

LUNG_IMAGE_SIZE = 224
INFECTION_IMAGE_SIZE = 256

LUNG_THRESHOLD = 0.30
INFECTION_THRESHOLD = 0.30

MASK_THRESHOLD = 128

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTED_LUNG_DIR = OUTPUT_DIR / "predicted_lung_masks"
PREDICTED_LUNG_DIR.mkdir(parents=True, exist_ok=True)


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
# IMAGE LOADING
# ============================================================

def load_grayscale_image(path, size):

    image = Image.open(path).convert("L")

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return image / 255.0


def load_binary_mask(path, size):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (size, size),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(mask)

    return (
        mask >= MASK_THRESHOLD
    ).astype(np.uint8)


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
# ANNOTATION GROUP
# ============================================================

def determine_annotation_group(reference, reference_lung):

    if np.array_equal(reference, reference_lung):
        return "IDENTICAL_TO_LUNG"

    lung_area = reference_lung.sum()

    if lung_area == 0:
        return "INVALID"

    ratio = reference.sum() / lung_area

    if ratio < 0.10:
        return "NON_IDENTICAL_SMALL"

    elif ratio < 0.50:
        return "NON_IDENTICAL_MODERATE"

    return "NON_IDENTICAL_LARGE"


# ============================================================
# LUNG MODEL PREDICTION
# ============================================================

def predict_lung_mask(lung_model, image_path):

    image = load_grayscale_image(
        image_path,
        LUNG_IMAGE_SIZE
    )

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    probability_map = lung_model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    # Apply the previously selected lung threshold.
    binary_mask = (
        probability_map >= LUNG_THRESHOLD
    ).astype(np.uint8)

    # Convert the 224x224 prediction to 256x256.
    # Nearest-neighbor interpolation preserves binary values.
    mask_image = Image.fromarray(
        binary_mask * 255
    )

    mask_image = mask_image.resize(
        (INFECTION_IMAGE_SIZE, INFECTION_IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    predicted_lung_mask = (
        np.asarray(mask_image) >= 128
    ).astype(np.uint8)

    return predicted_lung_mask


# ============================================================
# INFECTION MODEL PREDICTION
# ============================================================

def predict_infection_probability(infection_model, image_path):

    image = load_grayscale_image(
        image_path,
        INFECTION_IMAGE_SIZE
    )

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    probability_map = infection_model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    return probability_map


# ============================================================
# GLOBAL METRICS
# ============================================================

def calculate_global_metrics(counts):

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    dice_denominator = (2 * tp) + fp + fn
    iou_denominator = tp + fp + fn

    return {

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "global_dice": (
            (2 * tp) / dice_denominator
            if dice_denominator > 0 else 1.0
        ),

        "global_iou": (
            tp / iou_denominator
            if iou_denominator > 0 else 1.0
        ),

        "global_precision": (
            tp / (tp + fp)
            if (tp + fp) > 0 else 1.0
        ),

        "global_recall": (
            tp / (tp + fn)
            if (tp + fn) > 0 else 1.0
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AUTOMATIC LUNG-CONSTRAINED INFECTION SEGMENTATION")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"Lung input size: {LUNG_IMAGE_SIZE}")
    print(f"Infection input size: {INFECTION_IMAGE_SIZE}")
    print(f"Lung threshold: {LUNG_THRESHOLD}")
    print(f"Infection threshold: {INFECTION_THRESHOLD}")

    # --------------------------------------------------------
    # CHECK MODELS
    # --------------------------------------------------------

    if not LUNG_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Lung model not found: {LUNG_MODEL_PATH}"
        )

    if not INFECTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Infection model not found: {INFECTION_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print("\nLoading lung segmentation model...")

    lung_model = tf.keras.models.load_model(
        LUNG_MODEL_PATH,
        compile=False
    )

    print("Lung U-Net loaded successfully.")

    print("\nLoading infection segmentation model...")

    infection_model = tf.keras.models.load_model(
        INFECTION_MODEL_PATH,
        compile=False
    )

    print("Infection U-Net loaded successfully.")

    # --------------------------------------------------------
    # VALIDATION FILES
    # --------------------------------------------------------

    image_files = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
    ])

    print(
        f"\nValidation images found: {len(image_files)}"
    )

    # --------------------------------------------------------
    # METRIC ACCUMULATORS
    # --------------------------------------------------------

    methods = [
        "unconstrained",
        "reference_lung_constrained",
        "predicted_lung_constrained"
    ]

    global_counts = {
        method: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0
        }
        for method in methods
    }

    per_image_rows = []

    outside_prediction_pixels = 0
    total_unconstrained_prediction_pixels = 0

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

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

        # Reference masks are used only for evaluation.
        reference_infection = load_binary_mask(
            infection_path,
            INFECTION_IMAGE_SIZE
        )

        reference_lung = load_binary_mask(
            lung_path,
            INFECTION_IMAGE_SIZE
        )

        annotation_group = determine_annotation_group(
            reference_infection,
            reference_lung
        )

        if annotation_group == "INVALID":
            print(f"Skipping invalid sample: {stem}")
            continue

        # Generate automatic lung mask.
        predicted_lung = predict_lung_mask(
            lung_model,
            image_path
        )

        # Save predicted lung mask.
        predicted_lung_image = Image.fromarray(
            predicted_lung * 255
        )

        predicted_lung_image.save(
            PREDICTED_LUNG_DIR / f"{stem}_lung_prediction.png"
        )

        # Generate infection probabilities.
        infection_probability = predict_infection_probability(
            infection_model,
            image_path
        )

        # Original infection prediction.
        unconstrained_prediction = (
            infection_probability >= INFECTION_THRESHOLD
        ).astype(np.uint8)

        # Apply reference lung mask.
        reference_constrained_prediction = (
            unconstrained_prediction * reference_lung
        ).astype(np.uint8)

        # Apply automatically predicted lung mask.
        predicted_constrained_prediction = (
            unconstrained_prediction * predicted_lung
        ).astype(np.uint8)

        predictions = {

            "unconstrained":
                unconstrained_prediction,

            "reference_lung_constrained":
                reference_constrained_prediction,

            "predicted_lung_constrained":
                predicted_constrained_prediction

        }

        row = {

            "image_name": image_path.name,
            "annotation_group": annotation_group,

            "reference_lung_area":
                int(reference_lung.sum()),

            "predicted_lung_area":
                int(predicted_lung.sum())

        }

        # Lung segmentation quality against reference masks.
        lung_metrics = calculate_metrics(
            reference_lung,
            predicted_lung
        )

        row.update({

            "lung_dice": lung_metrics["dice"],
            "lung_iou": lung_metrics["iou"],
            "lung_precision": lung_metrics["precision"],
            "lung_recall": lung_metrics["recall"]

        })

        # Compare all three infection prediction methods.
        for method, prediction in predictions.items():

            metrics = calculate_metrics(
                reference_infection,
                prediction
            )

            for key in ["tp", "fp", "fn", "tn"]:

                global_counts[method][key] += (
                    metrics[key]
                )

            row.update({

                f"{method}_dice":
                    metrics["dice"],

                f"{method}_iou":
                    metrics["iou"],

                f"{method}_precision":
                    metrics["precision"],

                f"{method}_recall":
                    metrics["recall"],

                f"{method}_fp":
                    metrics["fp"],

                f"{method}_fn":
                    metrics["fn"],

                f"{method}_predicted_area":
                    int(prediction.sum())

            })

        # Difference between reference-constrained
        # and automatically constrained predictions.
        row["predicted_vs_reference_constraint_dice_difference"] = (
            row["predicted_lung_constrained_dice"]
            - row["reference_lung_constrained_dice"]
        )

        # Pixels outside the automatically predicted lung mask.
        outside_pixels = np.logical_and(
            unconstrained_prediction == 1,
            predicted_lung == 0
        ).sum()

        outside_prediction_pixels += int(outside_pixels)

        total_unconstrained_prediction_pixels += int(
            unconstrained_prediction.sum()
        )

        row["prediction_pixels_outside_predicted_lung"] = int(
            outside_pixels
        )

        per_image_rows.append(row)

        if index % 100 == 0 or index == len(image_files):

            print(
                f"Processed {index}/{len(image_files)} images"
            )

    # ========================================================
    # SAVE PER-IMAGE RESULTS
    # ========================================================

    results_df = pd.DataFrame(per_image_rows)

    per_image_path = (
        OUTPUT_DIR
        / "automatic_lung_constraint_per_image.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # ========================================================
    # MEAN PER-IMAGE SUMMARY
    # ========================================================

    mean_rows = []

    for method in methods:

        mean_rows.append({

            "method": method,

            "mean_dice":
                results_df[f"{method}_dice"].mean(),

            "mean_iou":
                results_df[f"{method}_iou"].mean(),

            "mean_precision":
                results_df[f"{method}_precision"].mean(),

            "mean_recall":
                results_df[f"{method}_recall"].mean(),

            "mean_fp":
                results_df[f"{method}_fp"].mean(),

            "mean_fn":
                results_df[f"{method}_fn"].mean()

        })

    mean_df = pd.DataFrame(mean_rows)

    mean_path = (
        OUTPUT_DIR
        / "automatic_lung_constraint_mean_summary.csv"
    )

    mean_df.to_csv(
        mean_path,
        index=False
    )

    # ========================================================
    # GLOBAL PIXEL SUMMARY
    # ========================================================

    global_rows = []

    for method in methods:

        metrics = calculate_global_metrics(
            global_counts[method]
        )

        global_rows.append({

            "method": method,
            **metrics

        })

    global_df = pd.DataFrame(global_rows)

    global_path = (
        OUTPUT_DIR
        / "automatic_lung_constraint_global_summary.csv"
    )

    global_df.to_csv(
        global_path,
        index=False
    )

    # ========================================================
    # SUBGROUP SUMMARY
    # ========================================================

    subgroup_rows = []

    for group, group_df in results_df.groupby(
        "annotation_group"
    ):

        subgroup_row = {

            "annotation_group": group,
            "sample_count": len(group_df)

        }

        for method in methods:

            subgroup_row.update({

                f"{method}_mean_dice":
                    group_df[f"{method}_dice"].mean(),

                f"{method}_mean_iou":
                    group_df[f"{method}_iou"].mean(),

                f"{method}_mean_precision":
                    group_df[f"{method}_precision"].mean(),

                f"{method}_mean_recall":
                    group_df[f"{method}_recall"].mean()

            })

        subgroup_rows.append(subgroup_row)

    subgroup_df = pd.DataFrame(subgroup_rows)

    subgroup_path = (
        OUTPUT_DIR
        / "automatic_lung_constraint_subgroup_summary.csv"
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False
    )

    # ========================================================
    # AUTOMATIC LUNG MODEL SUMMARY
    # ========================================================

    lung_summary = {

        "mean_dice":
            float(results_df["lung_dice"].mean()),

        "mean_iou":
            float(results_df["lung_iou"].mean()),

        "mean_precision":
            float(results_df["lung_precision"].mean()),

        "mean_recall":
            float(results_df["lung_recall"].mean())

    }

    # ========================================================
    # OUTSIDE-LUNG ANALYSIS
    # ========================================================

    outside_percentage = (
        100.0 * outside_prediction_pixels
        / total_unconstrained_prediction_pixels
        if total_unconstrained_prediction_pixels > 0 else 0.0
    )

    # ========================================================
    # JSON SUMMARY
    # ========================================================

    summary = {

        "validation_sample_count": len(results_df),

        "lung_threshold": LUNG_THRESHOLD,

        "infection_threshold": INFECTION_THRESHOLD,

        "lung_model_path": str(LUNG_MODEL_PATH),

        "infection_model_path": str(INFECTION_MODEL_PATH),

        "automatic_lung_model_metrics": lung_summary,

        "mean_per_image_metrics":
            mean_rows,

        "global_pixel_metrics":
            global_rows,

        "prediction_pixels_outside_predicted_lung":
            outside_prediction_pixels,

        "total_unconstrained_prediction_pixels":
            total_unconstrained_prediction_pixels,

        "percentage_prediction_pixels_outside_predicted_lung":
            outside_percentage,

        "interpretation_note":
            "Reference lung masks are used only as an evaluation "
            "comparison. Predicted lung masks are generated by the "
            "trained lung U-Net. Infection segmentation results "
            "are experimental and do not establish clinical validity."

    }

    json_path = (
        OUTPUT_DIR
        / "automatic_lung_constraint_summary.json"
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

    print("\n" + "=" * 80)
    print("AUTOMATIC LUNG SEGMENTATION QUALITY")
    print("=" * 80)

    print(
        pd.DataFrame([lung_summary]).round(4).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("MEAN PER-IMAGE INFECTION SEGMENTATION RESULTS")
    print("=" * 80)

    print(
        mean_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("GLOBAL PIXEL RESULTS")
    print("=" * 80)

    print(
        global_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("SUBGROUP RESULTS")
    print("=" * 80)

    print(
        subgroup_df.round(4).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("AUTOMATIC LUNG CONSTRAINT ANALYSIS")
    print("=" * 80)

    print(
        f"Prediction pixels outside predicted lungs: "
        f"{outside_prediction_pixels:,}"
    )

    print(
        f"Total unconstrained prediction pixels: "
        f"{total_unconstrained_prediction_pixels:,}"
    )

    print(
        f"Percentage outside predicted lungs: "
        f"{outside_percentage:.2f}%"
    )

    print("\nSaved files:")
    print(per_image_path)
    print(mean_path)
    print(global_path)
    print(subgroup_path)
    print(json_path)

    print("\nPredicted lung masks:")
    print(PREDICTED_LUNG_DIR)

    print("\nAUTOMATIC LUNG CONSTRAINT ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/102_evaluate_automatic_lung_constraint.py