
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_evaluation/validation_predictions.csv"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models/unet_lung_segmentation_balanced.keras"
)

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "saved_models/efficientnetb0_pneumonia_classifier_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification/lung_background_classifier_ablation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

MASK_THRESHOLD = 0.30

CLASS_NAMES = [
    "Normal",
    "Viral",
    "Bacterial",
    "COVID",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

CONDITIONS = [
    "original",
    "lung_only",
    "background_only",
]


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image_pair(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    # RGB image for EfficientNet
    rgb = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    rgb = tf.image.resize(
        rgb,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    # IMPORTANT:
    # EfficientNet input remains in the 0-255 range.
    rgb = tf.cast(rgb, tf.float32)

    # Grayscale image for U-Net
    grayscale = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    grayscale = tf.image.resize(
        grayscale,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    grayscale = tf.cast(grayscale, tf.float32) / 255.0

    return rgb.numpy(), grayscale.numpy()


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_classifier(model, images):

    probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    return np.asarray(probabilities)


def predict_unet(model, grayscale_images):

    probabilities = model.predict(
        grayscale_images,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    return np.asarray(probabilities)[..., 0]


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(y_true, y_pred):

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": np.mean(precision),
        "macro_recall": np.mean(recall),
        "macro_f1": np.mean(f1),
        "weighted_precision": precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )[0],
        "weighted_recall": precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )[1],
        "weighted_f1": precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )[2],
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("LUNG-ONLY VS BACKGROUND-ONLY CLASSIFIER ABLATION")
    print("=" * 65)

    # -----------------------------------------------------
    # Load validation records
    # -----------------------------------------------------

    print("\nLoading validation records...")

    df = pd.read_csv(VALIDATION_CSV)

    required_columns = [
        "processed_path",
        "actual_class",
        "predicted_class",
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

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("Validation CSV is empty.")

    invalid_classes = set(df["actual_class"]) - set(CLASS_NAMES)

    if invalid_classes:
        raise ValueError(
            f"Unknown actual classes: {invalid_classes}"
        )

    y_true = np.array([
        CLASS_TO_INDEX[name]
        for name in df["actual_class"]
    ])

    expected_predictions = np.array([
        CLASS_TO_INDEX[name]
        for name in df["predicted_class"]
    ])

    print(f"Validation images: {len(df)}")

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print("\nLoading balanced U-Net...")

    unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    print("Loading EfficientNetB0 classifier...")

    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH,
        compile=False
    )

    # -----------------------------------------------------
    # Prediction storage
    # -----------------------------------------------------

    all_predictions = {
        condition: []
        for condition in CONDITIONS
    }

    all_probabilities = {
        condition: []
        for condition in CONDITIONS
    }

    all_mask_areas = []

    # -----------------------------------------------------
    # Process in batches
    # -----------------------------------------------------

    total = len(df)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_df = df.iloc[start:end]

        rgb_batch = []
        grayscale_batch = []

        print(
            f"\nProcessing images {start + 1}-{end} of {total}"
        )

        for _, row in batch_df.iterrows():

            image_path = Path(row["processed_path"])

            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            rgb, grayscale = load_image_pair(image_path)

            rgb_batch.append(rgb)
            grayscale_batch.append(grayscale)

        rgb_batch = np.stack(rgb_batch).astype(np.float32)

        grayscale_batch = np.stack(
            grayscale_batch
        ).astype(np.float32)

        # -------------------------------------------------
        # Generate U-Net masks
        # -------------------------------------------------

        mask_probabilities = predict_unet(
            unet,
            grayscale_batch
        )

        masks = mask_probabilities >= MASK_THRESHOLD

        # -------------------------------------------------
        # Generate three classifier inputs
        # -------------------------------------------------

        lung_only_batch = (
            rgb_batch * masks[..., np.newaxis]
        )

        background_only_batch = (
            rgb_batch * (~masks)[..., np.newaxis]
        )

        # -------------------------------------------------
        # Original predictions
        # -------------------------------------------------

        original_probabilities = predict_classifier(
            classifier,
            rgb_batch
        )

        # -------------------------------------------------
        # Lung-only predictions
        # -------------------------------------------------

        lung_probabilities = predict_classifier(
            classifier,
            lung_only_batch
        )

        # -------------------------------------------------
        # Background-only predictions
        # -------------------------------------------------

        background_probabilities = predict_classifier(
            classifier,
            background_only_batch
        )

        batch_probabilities = {
            "original": original_probabilities,
            "lung_only": lung_probabilities,
            "background_only": background_probabilities,
        }

        for condition, probabilities in batch_probabilities.items():

            predictions = np.argmax(
                probabilities,
                axis=1
            )

            all_probabilities[condition].extend(
                probabilities.tolist()
            )

            all_predictions[condition].extend(
                predictions.tolist()
            )

        mask_areas = np.mean(
            masks,
            axis=(1, 2)
        ) * 100

        all_mask_areas.extend(
            mask_areas.tolist()
        )

    # -----------------------------------------------------
    # Convert predictions
    # -----------------------------------------------------

    print("\nAll validation images processed.")

    for condition in CONDITIONS:

        all_predictions[condition] = np.asarray(
            all_predictions[condition]
        )

        all_probabilities[condition] = np.asarray(
            all_probabilities[condition]
        )

    # -----------------------------------------------------
    # Baseline verification
    # -----------------------------------------------------

    print("\nVerifying original-image predictions...")

    baseline_predictions = all_predictions["original"]

    agreement = np.mean(
        baseline_predictions == expected_predictions
    )

    mean_probability_difference = np.mean(
        np.abs(
            all_probabilities["original"]
            - df[
                [
                    "probability_Normal",
                    "probability_Viral",
                    "probability_Bacterial",
                    "probability_COVID",
                ]
            ].to_numpy(dtype=np.float32)
        )
    )

    print(
        f"Prediction agreement: {agreement * 100:.4f}%"
    )

    print(
        "Mean probability difference: "
        f"{mean_probability_difference:.10f}"
    )

    baseline_report = {
        "validation_images": total,
        "prediction_agreement_percent": agreement * 100,
        "mean_probability_difference": mean_probability_difference,
        "mask_threshold": MASK_THRESHOLD,
    }

    pd.DataFrame([baseline_report]).to_csv(
        OUTPUT_DIR / "baseline_verification.csv",
        index=False
    )

    if agreement < 0.999:
        raise RuntimeError(
            "Baseline verification failed. "
            "Original predictions do not sufficiently match "
            "the existing validation predictions. "
            "Do not interpret the ablation results."
        )

    print("Baseline verification passed.")

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    summary_rows = []

    for condition in CONDITIONS:

        metrics = calculate_metrics(
            y_true,
            all_predictions[condition]
        )

        metrics["condition"] = condition

        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df[
        [
            "condition",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ]
    ]

    summary_df.to_csv(
        OUTPUT_DIR / "ablation_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # Class-wise reports and confusion matrices
    # -----------------------------------------------------

    reports = {}
    confusion_matrices = {}

    for condition in CONDITIONS:

        predictions = all_predictions[condition]

        reports[condition] = classification_report(
            y_true,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        confusion_matrices[condition] = confusion_matrix(
            y_true,
            predictions,
            labels=list(range(len(CLASS_NAMES)))
        ).tolist()

    import json

    with open(
        OUTPUT_DIR / "classification_reports.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            reports,
            file,
            indent=4
        )

    with open(
        OUTPUT_DIR / "confusion_matrices.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            confusion_matrices,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Source-wise accuracy
    # -----------------------------------------------------

    source_rows = []

    source_values = df["source_dataset"].to_numpy()

    for condition in CONDITIONS:

        predictions = all_predictions[condition]

        for source in sorted(df["source_dataset"].dropna().unique()):

            source_mask = source_values == source

            source_rows.append({
                "condition": condition,
                "source_dataset": source,
                "total_images": int(np.sum(source_mask)),
                "accuracy": accuracy_score(
                    y_true[source_mask],
                    predictions[source_mask]
                ),
            })

    pd.DataFrame(source_rows).to_csv(
        OUTPUT_DIR / "source_wise_accuracy.csv",
        index=False
    )

    # -----------------------------------------------------
    # Save detailed predictions
    # -----------------------------------------------------

    detailed_df = df.copy()

    detailed_df["unet_mask_area_percent"] = all_mask_areas

    for condition in CONDITIONS:

        detailed_df[f"{condition}_predicted_class"] = [
            CLASS_NAMES[index]
            for index in all_predictions[condition]
        ]

        detailed_df[f"{condition}_correct"] = (
            all_predictions[condition] == y_true
        )

        for class_index, class_name in enumerate(CLASS_NAMES):

            detailed_df[
                f"{condition}_probability_{class_name}"
            ] = all_probabilities[condition][:, class_index]

    detailed_df.to_csv(
        OUTPUT_DIR / "ablation_predictions.csv",
        index=False
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("ABLATION SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nMask area statistics:")

    print(
        pd.Series(all_mask_areas).describe().to_string()
    )

    print("\nClass-wise classification reports:")

    for condition in CONDITIONS:

        print(f"\n--- {condition.upper()} ---")

        print(
            classification_report(
                y_true,
                all_predictions[condition],
                labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

    print("\nResults saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- baseline_verification.csv")
    print("- ablation_summary.csv")
    print("- classification_reports.json")
    print("- confusion_matrices.json")
    print("- source_wise_accuracy.csv")
    print("- ablation_predictions.csv")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/19_run_lung_background_classifier_ablation.py