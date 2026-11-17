
"""
57_lung_only_classifier_ablation.py

Purpose:
Compare original chest X-ray classification with classification
after replacing pixels outside the U-Net lung mask with per-channel
image mean values.

Uses:
- Original EfficientNetB0 checkpoint
- Balanced U-Net checkpoint
- Existing validation split only
- Existing validation predictions as baseline verification

Does NOT:
- Train or modify either model
- Access the test split
- Tune thresholds
- Calculate infection percentage
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "validation_evaluation"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "lung_only_classifier_ablation"
)

IMAGE_SIZE = 224
LUNG_THRESHOLD = 0.30
VIRAL_THRESHOLD = 0.25

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 16
BASELINE_TOLERANCE = 0.0001


# ============================================================
# PREPROCESSING
# Keep identical to script 56
# ============================================================

def load_rgb_image(image_path):
    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )

    # EfficientNet preprocessing: keep values in 0..255.
    return tf.cast(image, tf.float32).numpy()


def load_grayscale_image(image_path):
    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    image.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )

    # U-Net preprocessing: normalize to 0..1.
    return (
        tf.cast(image, tf.float32) / 255.0
    ).numpy()


# ============================================================
# DECISION RULES
# ============================================================

def apply_viral_threshold(probabilities):
    """
    Existing validation-selected decision rule:
    If Viral probability >= 0.25, predict Viral.
    Otherwise choose the highest non-Viral probability.
    """
    predictions = []

    for row in probabilities:
        viral_probability = float(row[1])

        if viral_probability >= VIRAL_THRESHOLD:
            predictions.append(1)
        else:
            non_viral_indices = [0, 2, 3]

            selected = max(
                non_viral_indices,
                key=lambda index: float(row[index]),
            )

            predictions.append(selected)

    return np.asarray(predictions, dtype=np.int32)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        ).tolist(),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("LUNG-ONLY CLASSIFIER ABLATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    for path in [
        CLASSIFIER_PATH,
        UNET_PATH,
        PREDICTIONS_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load validation predictions
    # --------------------------------------------------------

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "processed_path",
        "actual_class",
        "predicted_class",
        *[
            f"probability_{name}"
            for name in CLASS_NAMES
        ],
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Validation CSV is empty.")

    if not set(df["actual_class"]).issubset(CLASS_NAMES):
        raise ValueError("Unknown actual class in validation CSV.")

    if not set(df["predicted_class"]).issubset(CLASS_NAMES):
        raise ValueError("Unknown predicted class in validation CSV.")

    image_paths = df["processed_path"].tolist()

    missing_images = [
        path
        for path in image_paths
        if not Path(path).is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} validation images are missing. "
            f"First missing path: {missing_images[0]}"
        )

    y_true = np.asarray(
        [
            CLASS_NAMES.index(name)
            for name in df["actual_class"]
        ],
        dtype=np.int32,
    )

    saved_probabilities = df[
        [
            f"probability_{name}"
            for name in CLASS_NAMES
        ]
    ].to_numpy(dtype=np.float32)

    saved_predictions = np.asarray(
        [
            CLASS_NAMES.index(name)
            for name in df["predicted_class"]
        ],
        dtype=np.int32,
    )

    print(f"\nValidation images: {len(df)}")

    print("\nClass distribution:")
    print(df["actual_class"].value_counts().to_string())

    # --------------------------------------------------------
    # Load existing models
    # --------------------------------------------------------

    print("\nLoading original EfficientNetB0...")
    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH,
        compile=False,
    )

    print("Loading balanced U-Net...")
    unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False,
    )

    # --------------------------------------------------------
    # Process validation images in batches
    # --------------------------------------------------------

    original_probability_batches = []
    lung_only_probability_batches = []
    mask_area_percentages = []

    for start in range(0, len(image_paths), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(image_paths),
        )

        batch_paths = image_paths[start:end]

        print(
            f"\nProcessing images {start + 1}–{end} "
            f"of {len(image_paths)}"
        )

        rgb_images = []
        gray_images = []

        for image_path in batch_paths:

            rgb_images.append(
                load_rgb_image(image_path)
            )

            gray_images.append(
                load_grayscale_image(image_path)
            )

        rgb_batch = np.stack(
            rgb_images
        ).astype(np.float32)

        gray_batch = np.stack(
            gray_images
        ).astype(np.float32)

        # ----------------------------------------------------
        # U-Net lung segmentation
        # ----------------------------------------------------

        mask_probabilities = unet(
            gray_batch,
            training=False,
        ).numpy()

        if (
            mask_probabilities.ndim != 4
            or mask_probabilities.shape[-1] < 1
        ):
            raise ValueError(
                "Unexpected U-Net output shape: "
                f"{mask_probabilities.shape}"
            )

        lung_masks = (
            mask_probabilities[..., 0]
            >= LUNG_THRESHOLD
        )

        # ----------------------------------------------------
        # Original image classification
        # ----------------------------------------------------

        original_probabilities = classifier(
            rgb_batch,
            training=False,
        ).numpy()

        # ----------------------------------------------------
        # Construct lung-only images
        # ----------------------------------------------------

        lung_only_batch = np.empty_like(
            rgb_batch,
            dtype=np.float32,
        )

        for index in range(len(rgb_batch)):

            image = rgb_batch[index]
            mask = lung_masks[index]

            # Per-channel mean of the original image.
            channel_mean = np.mean(
                image,
                axis=(0, 1),
                keepdims=True,
            )

            # Preserve lung pixels; replace everything outside.
            lung_only_batch[index] = np.where(
                mask[..., np.newaxis],
                image,
                channel_mean,
            )

            mask_area = (
                np.count_nonzero(mask)
                / mask.size
            ) * 100.0

            mask_area_percentages.append(
                float(mask_area)
            )

        # ----------------------------------------------------
        # Lung-only classification
        # ----------------------------------------------------

        lung_only_probabilities = classifier(
            lung_only_batch,
            training=False,
        ).numpy()

        if (
            original_probabilities.shape
            != (len(batch_paths), NUM_CLASSES)
        ):
            raise ValueError(
                "Unexpected original classifier output shape: "
                f"{original_probabilities.shape}"
            )

        if (
            lung_only_probabilities.shape
            != (len(batch_paths), NUM_CLASSES)
        ):
            raise ValueError(
                "Unexpected lung-only classifier output shape: "
                f"{lung_only_probabilities.shape}"
            )

        original_probability_batches.append(
            original_probabilities
        )

        lung_only_probability_batches.append(
            lung_only_probabilities
        )

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    original_probabilities = np.concatenate(
        original_probability_batches,
        axis=0,
    )

    lung_only_probabilities = np.concatenate(
        lung_only_probability_batches,
        axis=0,
    )

    mask_area_percentages = np.asarray(
        mask_area_percentages,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Verify original model against saved baseline
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VERIFYING ORIGINAL CLASSIFIER BASELINE")
    print("=" * 65)

    absolute_differences = np.abs(
        original_probabilities - saved_probabilities
    )

    mean_absolute_difference = float(
        np.mean(absolute_differences)
    )

    max_absolute_difference = float(
        np.max(absolute_differences)
    )

    original_argmax = np.argmax(
        original_probabilities,
        axis=1,
    )

    baseline_prediction_agreement = float(
        np.mean(
            original_argmax == saved_predictions
        )
    )

    baseline_verified = (
        np.allclose(
            original_probabilities,
            saved_probabilities,
            atol=BASELINE_TOLERANCE,
            rtol=0,
        )
        and np.array_equal(
            original_argmax,
            saved_predictions,
        )
    )

    baseline_verification = {
        "number_of_images": int(len(df)),
        "baseline_verified": bool(baseline_verified),
        "probability_tolerance": BASELINE_TOLERANCE,
        "mean_absolute_probability_difference": (
            mean_absolute_difference
        ),
        "maximum_absolute_probability_difference": (
            max_absolute_difference
        ),
        "prediction_agreement": (
            baseline_prediction_agreement
        ),
    }

    with open(
        OUTPUT_DIR / "baseline_verification.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            baseline_verification,
            file,
            indent=4,
        )

    print(
        "Mean absolute probability difference:",
        mean_absolute_difference,
    )

    print(
        "Maximum absolute probability difference:",
        max_absolute_difference,
    )

    print(
        "Prediction agreement:",
        f"{baseline_prediction_agreement * 100:.3f}%",
    )

    print(
        "Baseline verified:",
        baseline_verified,
    )

    if not baseline_verified:
        raise RuntimeError(
            "Original classifier does not reproduce the saved "
            "validation baseline within tolerance. "
            "Lung-only results will not be evaluated. "
            "Check preprocessing, model checkpoint, and prediction CSV."
        )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    original_argmax = np.argmax(
        original_probabilities,
        axis=1,
    )

    lung_only_argmax = np.argmax(
        lung_only_probabilities,
        axis=1,
    )

    original_threshold_predictions = (
        apply_viral_threshold(
            original_probabilities
        )
    )

    lung_only_threshold_predictions = (
        apply_viral_threshold(
            lung_only_probabilities
        )
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    print("\nCalculating classification metrics...")

    metrics = {
        "original_argmax": calculate_metrics(
            y_true,
            original_argmax,
        ),
        "lung_only_argmax": calculate_metrics(
            y_true,
            lung_only_argmax,
        ),
        "original_viral_threshold_025": calculate_metrics(
            y_true,
            original_threshold_predictions,
        ),
        "lung_only_viral_threshold_025": calculate_metrics(
            y_true,
            lung_only_threshold_predictions,
        ),
    }

    # --------------------------------------------------------
    # Probability and prediction changes
    # --------------------------------------------------------

    probability_changes = (
        lung_only_probabilities
        - original_probabilities
    )

    changed_argmax = (
        original_argmax != lung_only_argmax
    )

    changed_threshold_prediction = (
        original_threshold_predictions
        != lung_only_threshold_predictions
    )

    # --------------------------------------------------------
    # Per-image output
    # --------------------------------------------------------

    results_df = df[
        [
            "processed_path",
            "actual_class",
            "source_dataset",
        ]
    ].copy()

    results_df["original_argmax_prediction"] = [
        CLASS_NAMES[index]
        for index in original_argmax
    ]

    results_df["lung_only_argmax_prediction"] = [
        CLASS_NAMES[index]
        for index in lung_only_argmax
    ]

    results_df["original_threshold_prediction"] = [
        CLASS_NAMES[index]
        for index in original_threshold_predictions
    ]

    results_df["lung_only_threshold_prediction"] = [
        CLASS_NAMES[index]
        for index in lung_only_threshold_predictions
    ]

    results_df["original_argmax_correct"] = (
        original_argmax == y_true
    )

    results_df["lung_only_argmax_correct"] = (
        lung_only_argmax == y_true
    )

    results_df["argmax_prediction_changed"] = (
        changed_argmax
    )

    results_df["threshold_prediction_changed"] = (
        changed_threshold_prediction
    )

    results_df["lung_mask_area_percent"] = (
        mask_area_percentages
    )

    for index, class_name in enumerate(CLASS_NAMES):

        results_df[
            f"original_probability_{class_name}"
        ] = original_probabilities[:, index]

        results_df[
            f"lung_only_probability_{class_name}"
        ] = lung_only_probabilities[:, index]

        results_df[
            f"probability_change_{class_name}"
        ] = probability_changes[:, index]

    results_df.to_csv(
        OUTPUT_DIR / "lung_only_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Mask quality statistics
    # --------------------------------------------------------

    mask_statistics = {
        "mean_lung_mask_area_percent": float(
            np.mean(mask_area_percentages)
        ),
        "median_lung_mask_area_percent": float(
            np.median(mask_area_percentages)
        ),
        "minimum_lung_mask_area_percent": float(
            np.min(mask_area_percentages)
        ),
        "maximum_lung_mask_area_percent": float(
            np.max(mask_area_percentages)
        ),
        "empty_masks": int(
            np.sum(mask_area_percentages == 0)
        ),
        "masks_below_1_percent": int(
            np.sum(mask_area_percentages < 1)
        ),
        "masks_below_5_percent": int(
            np.sum(mask_area_percentages < 5)
        ),
        "masks_below_10_percent": int(
            np.sum(mask_area_percentages < 10)
        ),
    }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": "lung_only_classifier_ablation",
        "validation_image_count": int(len(df)),
        "class_names": CLASS_NAMES,
        "classifier_checkpoint": str(CLASSIFIER_PATH),
        "unet_checkpoint": str(UNET_PATH),
        "lung_threshold": LUNG_THRESHOLD,
        "viral_threshold": VIRAL_THRESHOLD,
        "batch_size": BATCH_SIZE,
        "baseline_verification": baseline_verification,
        "mask_statistics": mask_statistics,
        "prediction_comparison": {
            "argmax_prediction_changes": int(
                np.sum(changed_argmax)
            ),
            "argmax_prediction_change_percent": float(
                np.mean(changed_argmax) * 100
            ),
            "threshold_prediction_changes": int(
                np.sum(changed_threshold_prediction)
            ),
            "threshold_prediction_change_percent": float(
                np.mean(changed_threshold_prediction) * 100
            ),
            "mean_absolute_probability_change": float(
                np.mean(np.abs(probability_changes))
            ),
        },
        "metrics": metrics,
        "interpretation_limitations": [
            "U-Net masks represent lung regions, not infection regions.",
            "Lung-only masking is an intervention and may introduce "
            "artificial image boundaries.",
            "Changes in predictions do not independently prove "
            "shortcut learning.",
            "Poor lung segmentation can affect lung-only results.",
            "This experiment uses validation data only.",
            "No model training or threshold tuning is performed.",
        ],
    }

    with open(
        OUTPUT_DIR / "lung_only_ablation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Compact metrics CSV
    # --------------------------------------------------------

    metric_rows = []

    for experiment_name, values in metrics.items():

        metric_rows.append({
            "experiment": experiment_name,
            "accuracy": values["accuracy"],
            "macro_precision": values["macro_precision"],
            "macro_recall": values["macro_recall"],
            "macro_f1": values["macro_f1"],
            "weighted_f1": values["weighted_f1"],
        })

    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "lung_only_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save confusion matrices and reports
    # --------------------------------------------------------

    detailed_results = {
        name: {
            "confusion_matrix": values["confusion_matrix"],
            "classification_report": values[
                "classification_report"
            ],
        }
        for name, values in metrics.items()
    }

    with open(
        OUTPUT_DIR / "lung_only_detailed_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            detailed_results,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("LUNG-ONLY ABLATION RESULTS")
    print("=" * 65)

    print(
        pd.DataFrame(metric_rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nPrediction changes:")
    print(
        "Argmax:",
        int(np.sum(changed_argmax)),
        "/",
        len(df),
    )

    print(
        "Viral-threshold:",
        int(np.sum(changed_threshold_prediction)),
        "/",
        len(df),
    )

    print("\nLung mask statistics:")
    for key, value in mask_statistics.items():
        print(f"{key}: {value}")

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nExperiment completed successfully.")
    print("No model weights were modified.")
    print("The held-out test set was not accessed.")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/57_lung_only_classifier_ablation.py