
"""
SCRIPT 109
Independent Test Evaluation of Selected Infection Threshold

Compares:
- Original threshold: 0.50
- Validation-selected threshold: 0.30

Uses:
- COVID infection U-Net
- Fine-tuned lung U-Net
- Automatically predicted lung masks

Test dataset: 583 COVID images.

No training or threshold selection is performed.
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

TEST_ROOT = (
    ROOT / "data" / "raw" / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Test" / "COVID-19"
)

IMAGE_DIR = TEST_ROOT / "images"
REFERENCE_INFECTION_DIR = TEST_ROOT / "infection masks"

INFECTION_MODEL_PATH = (
    ROOT / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

LUNG_MODEL_PATH = (
    ROOT / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

OUTPUT_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "selected_threshold_independent_test_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INFECTION_SIZE = 256
LUNG_SIZE = 224
BATCH_SIZE = 8

INFECTION_THRESHOLD = 0.30
BASELINE_THRESHOLD = 0.50
LUNG_THRESHOLD = 0.30


# ============================================================
# FILE LOOKUP
# ============================================================

VALID_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def build_lookup(directory):

    if not directory.exists():
        raise FileNotFoundError(directory)

    lookup = {}

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        lookup[path.stem.lower()] = path

    return lookup


# ============================================================
# PREPROCESSING
# ============================================================

def load_grayscale(path, size):

    image = Image.open(path).convert("L")

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0


def load_binary_mask(path, size):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (size, size),
        Image.Resampling.NEAREST,
    )

    return np.asarray(mask) >= 128


def resize_binary_mask(mask, size):

    resized = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    )

    resized = resized.resize(
        (size, size),
        Image.Resampling.NEAREST,
    )

    return np.asarray(resized) >= 128


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(reference, prediction):

    tp = int(np.logical_and(reference, prediction).sum())
    fp = int(np.logical_and(~reference, prediction).sum())
    fn = int(np.logical_and(reference, ~prediction).sum())
    tn = int(np.logical_and(~reference, ~prediction).sum())

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 1.0
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INDEPENDENT TEST EVALUATION")
    print("=" * 80)

    image_lookup = build_lookup(IMAGE_DIR)
    reference_lookup = build_lookup(
        REFERENCE_INFECTION_DIR
    )

    image_names = sorted(
        set(image_lookup) & set(reference_lookup)
    )

    print(f"\nMatched test images: {len(image_names)}")

    if len(image_names) != 583:
        raise ValueError(
            f"Expected 583 COVID test images, found {len(image_names)}"
        )

    print("\nLoading infection U-Net...")
    infection_model = tf.keras.models.load_model(
        INFECTION_MODEL_PATH,
        compile=False,
    )

    print("Loading fine-tuned lung U-Net...")
    lung_model = tf.keras.models.load_model(
        LUNG_MODEL_PATH,
        compile=False,
    )

    print("Both models loaded.")

    image_batch = []
    reference_masks = []
    original_images = []

    for index, name in enumerate(image_names):

        image_path = image_lookup[name]
        reference_path = reference_lookup[name]

        infection_image = load_grayscale(
            image_path,
            INFECTION_SIZE,
        )

        reference = load_binary_mask(
            reference_path,
            INFECTION_SIZE,
        )

        image_batch.append(
            infection_image[..., np.newaxis]
        )

        reference_masks.append(reference)

        # Keep the original image for lung-model preprocessing.
        lung_image = load_grayscale(
            image_path,
            LUNG_SIZE,
        )

        original_images.append(
            lung_image[..., np.newaxis]
        )

        if (index + 1) % 100 == 0:
            print(f"Loaded {index + 1}/583 images")

    image_batch = np.asarray(
        image_batch,
        dtype=np.float32,
    )

    original_images = np.asarray(
        original_images,
        dtype=np.float32,
    )

    reference_masks = np.asarray(
        reference_masks,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Infection probability maps.
    # --------------------------------------------------------

    print("\nPredicting infection probability maps...")

    infection_predictions = []

    for start in range(0, len(image_batch), BATCH_SIZE):

        batch = image_batch[start:start + BATCH_SIZE]

        predictions = infection_model.predict(
            batch,
            verbose=0,
        )

        infection_predictions.append(predictions)

    infection_predictions = np.concatenate(
        infection_predictions,
        axis=0,
    )[..., 0]

    # --------------------------------------------------------
    # Automatic lung masks.
    # --------------------------------------------------------

    print("\nPredicting automatic lung masks...")

    lung_predictions = []

    for start in range(0, len(original_images), BATCH_SIZE):

        batch = original_images[start:start + BATCH_SIZE]

        predictions = lung_model.predict(
            batch,
            verbose=0,
        )

        lung_predictions.append(predictions)

    lung_predictions = np.concatenate(
        lung_predictions,
        axis=0,
    )[..., 0]

    # --------------------------------------------------------
    # Resize lung masks to infection resolution.
    # --------------------------------------------------------

    print("\nPreparing automatic lung constraints...")

    lung_masks = []

    for probability_map in lung_predictions:

        binary = probability_map >= LUNG_THRESHOLD

        resized = resize_binary_mask(
            binary,
            INFECTION_SIZE,
        )

        lung_masks.append(resized)

    lung_masks = np.asarray(
        lung_masks,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Evaluate both thresholds.
    # --------------------------------------------------------

    rows = []

    thresholds = [
        ("BASELINE_0.50", BASELINE_THRESHOLD),
        ("VALIDATION_SELECTED_0.30", INFECTION_THRESHOLD),
    ]

    for index, image_name in enumerate(image_names):

        reference = reference_masks[index]

        probability = infection_predictions[index]
        lung = lung_masks[index]

        for threshold_name, threshold in thresholds:

            unconstrained = probability >= threshold

            constrained = unconstrained & lung

            for strategy, prediction in [
                ("UNCONSTRAINED", unconstrained),
                ("AUTOMATIC_LUNG_CONSTRAINED", constrained),
            ]:

                metrics = calculate_metrics(
                    reference,
                    prediction,
                )

                rows.append({
                    "image_name": image_name,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "strategy": strategy,
                    **metrics,
                })

        if (index + 1) % 100 == 0:
            print(
                f"Evaluated {index + 1}/583 images"
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_DIR / "test_per_image_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    summary_rows = []

    for (threshold_name, threshold, strategy), group in (
        results.groupby(
            ["threshold_name", "threshold", "strategy"]
        )
    ):

        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())

        summary_rows.append({
            "threshold_name": threshold_name,
            "threshold": float(threshold),
            "strategy": strategy,
            "image_count": int(len(group)),
            "mean_image_dice": float(group["dice"].mean()),
            "median_image_dice": float(group["dice"].median()),
            "global_pixel_dice": float(
                2 * tp / (2 * tp + fp + fn)
            ),
            "global_pixel_iou": float(
                tp / (tp + fp + fn)
            ),
            "global_precision": float(
                tp / (tp + fp) if tp + fp else 1.0
            ),
            "global_recall": float(
                tp / (tp + fn) if tp + fn else 1.0
            ),
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "test_threshold_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("INDEPENDENT TEST RESULTS")
    print("=" * 80)

    print(summary_df.to_string(index=False))

    summary_json = {
        "test_images": len(image_names),
        "baseline_threshold": BASELINE_THRESHOLD,
        "validation_selected_threshold": INFECTION_THRESHOLD,
        "lung_threshold": LUNG_THRESHOLD,
        "results": summary_rows,
        "limitations": [
            "Threshold 0.30 was selected using validation data.",
            "The test set was not used for threshold selection.",
            "The test results provide an independent evaluation.",
            "The experiment is limited to COVID infection segmentation.",
            "This is an experimental research model, not a clinical tool.",
        ],
    }

    with open(
        OUTPUT_DIR / "test_threshold_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_json,
            file,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("SCRIPT 109 COMPLETED")
    print("=" * 80)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated:")
    print("1. test_per_image_metrics.csv")
    print("2. test_threshold_comparison.csv")
    print("3. test_threshold_summary.json")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/109_evaluate_selected_threshold_on_test.py