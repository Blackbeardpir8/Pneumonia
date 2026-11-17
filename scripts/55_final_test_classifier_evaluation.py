"""
Final held-out test evaluation for the Pneumonia CDSS EfficientNetB0 candidates.

IMPORTANT:
- Uses ONLY rows where manifest.split == "test".
- Does not train, fine-tune, calibrate, or change either model.
- Applies the validation-selected Viral decision thresholds:
    original model: 0.250
    source-balanced model: 0.275
- Uses the same TensorFlow image decoding/resizing approach used in prior
  classifier experiments: RGB, bilinear resize to 224x224, float32 0..255.
- Writes reports and predictions under results/classification/final_test_evaluation/.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/processed/classification/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/classification/final_test_evaluation"

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

CANDIDATES = [
    {
        "name": "original_threshold_0_250",
        "model_path": PROJECT_ROOT / "saved_models/efficientnetb0_pneumonia_classifier_best.keras",
        "viral_threshold": 0.250,
    },
    {
        "name": "source_balanced_threshold_0_275",
        "model_path": PROJECT_ROOT / "saved_models/efficientnetb0_source_balanced_best.keras",
        "viral_threshold": 0.275,
    },
]

# ---------------------------------------------------------------------
# Data validation and preprocessing
# ---------------------------------------------------------------------

def load_test_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"split", "class_name", "class_index", "processed_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    test_df = manifest[manifest["split"].astype(str).str.lower() == "test"].copy()
    if test_df.empty:
        raise ValueError("No rows with split='test' were found in the manifest.")

    test_df["class_name"] = test_df["class_name"].astype(str)
    test_df["class_index"] = pd.to_numeric(test_df["class_index"], errors="raise").astype(int)
    test_df["processed_path"] = test_df["processed_path"].astype(str)

    unexpected_classes = sorted(set(test_df["class_name"]) - set(CLASS_NAMES))
    if unexpected_classes:
        raise ValueError(f"Unexpected class names in test manifest: {unexpected_classes}")

    expected_mapping = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    wrong_mapping = test_df[
        test_df.apply(
            lambda row: expected_mapping[row["class_name"]] != row["class_index"],
            axis=1,
        )
    ]
    if not wrong_mapping.empty:
        raise ValueError(
            "class_name/class_index mismatch found in test manifest. "
            "Check CLASS_NAMES and manifest mapping before evaluation."
        )

    paths = []
    missing_files = []
    for value in test_df["processed_path"]:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        paths.append(str(path))
        if not path.is_file():
            missing_files.append(str(path))

    if missing_files:
        preview = "\n".join(missing_files[:20])
        raise FileNotFoundError(
            f"{len(missing_files)} test image file(s) are missing. First paths:\n{preview}"
        )

    test_df["absolute_path"] = paths
    return test_df.reset_index(drop=True)


def decode_and_resize(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
    )
    image = tf.cast(image, tf.float32)  # Keep 0..255; EfficientNet has internal rescaling.
    return image, label


def make_test_dataset(test_df):
    paths = test_df["absolute_path"].values
    labels = test_df["class_index"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# Metrics and decision rule
# ---------------------------------------------------------------------

def apply_viral_threshold(probabilities, threshold):
    """
    If P(Viral) >= threshold, predict Viral.
    Otherwise choose the highest-probability non-Viral class.
    """
    non_viral_indices = [i for i in range(NUM_CLASSES) if i != 1]
    non_viral_predictions = np.argmax(
        probabilities[:, non_viral_indices],
        axis=1,
    )
    predictions = np.asarray(non_viral_indices, dtype=np.int32)[non_viral_predictions]
    viral_mask = probabilities[:, 1] >= threshold
    predictions[viral_mask] = 1
    return predictions


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate_candidate(candidate, dataset, test_df):
    model_path = candidate["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("\n" + "=" * 72)
    print(f"Evaluating: {candidate['name']}")
    print(f"Checkpoint: {model_path}")
    print(f"Viral threshold: {candidate['viral_threshold']}")
    print("=" * 72)

    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(dataset, verbose=1)

    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected model output shape (N, {NUM_CLASSES}), "
            f"received {probabilities.shape}."
        )
    if len(probabilities) != len(test_df):
        raise ValueError(
            f"Prediction count {len(probabilities)} does not match "
            f"test manifest count {len(test_df)}."
        )

    y_true = test_df["class_index"].to_numpy(dtype=np.int32)
    y_argmax = np.argmax(probabilities, axis=1)
    y_threshold = apply_viral_threshold(
        probabilities,
        candidate["viral_threshold"],
    )

    argmax_metrics = calculate_metrics(y_true, y_argmax)
    threshold_metrics = calculate_metrics(y_true, y_threshold)

    prefix = candidate["name"]
    prediction_df = test_df[
        ["processed_path", "class_name", "class_index"]
        + (["source_dataset"] if "source_dataset" in test_df.columns else [])
    ].copy()
    prediction_df = prediction_df.rename(
        columns={
            "class_name": "actual_class",
            "class_index": "actual_class_index",
        }
    )

    for index, class_name in enumerate(CLASS_NAMES):
        prediction_df[f"prob_{class_name.lower()}"] = probabilities[:, index]

    prediction_df["argmax_predicted_class_index"] = y_argmax
    prediction_df["argmax_predicted_class"] = [
        CLASS_NAMES[index] for index in y_argmax
    ]
    prediction_df["threshold_predicted_class_index"] = y_threshold
    prediction_df["threshold_predicted_class"] = [
        CLASS_NAMES[index] for index in y_threshold
    ]
    prediction_df.to_csv(
        OUTPUT_DIR / f"{prefix}_predictions.csv",
        index=False,
    )

    for rule_name, metrics in [
        ("argmax", argmax_metrics),
        ("viral_threshold", threshold_metrics),
    ]:
        cm_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
        cm_df.index.name = "actual_class"
        cm_df.to_csv(OUTPUT_DIR / f"{prefix}_{rule_name}_confusion_matrix.csv")

        with open(
            OUTPUT_DIR / f"{prefix}_{rule_name}_metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                file,
                indent=4,
            )

    print("\nARGMAX METRICS")
    print(classification_report(
        y_true, y_argmax,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Argmax confusion matrix (rows=actual, columns=predicted):")
    print(argmax_metrics["confusion_matrix"])

    print("\nVIRAL-THRESHOLD METRICS")
    print(classification_report(
        y_true, y_threshold,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Threshold confusion matrix (rows=actual, columns=predicted):")
    print(threshold_metrics["confusion_matrix"])

    del model
    return {
        "candidate": candidate["name"],
        "checkpoint": str(model_path.relative_to(PROJECT_ROOT)),
        "viral_threshold": candidate["viral_threshold"],
        "argmax": {
            key: value
            for key, value in argmax_metrics.items()
            if key != "confusion_matrix"
        },
        "viral_threshold_rule": {
            key: value
            for key, value in threshold_metrics.items()
            if key != "confusion_matrix"
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("Project root:", PROJECT_ROOT)
    print("Manifest:", MANIFEST_PATH)
    print("\nLoading held-out test manifest...")

    test_df = load_test_manifest()

    print("Test image count:", len(test_df))
    print("\nTest class counts:")
    print(test_df["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    if "source_dataset" in test_df.columns:
        print("\nTest source counts:")
        print(test_df["source_dataset"].value_counts(dropna=False))

    dataset = make_test_dataset(test_df)

    all_results = []
    for candidate in CANDIDATES:
        all_results.append(evaluate_candidate(candidate, dataset, test_df))

    summary_rows = []
    for result in all_results:
        for rule_key, rule_label in [
            ("argmax", "argmax"),
            ("viral_threshold_rule", "viral_threshold"),
        ]:
            metrics = result[rule_key]
            summary_rows.append({
                "candidate": result["candidate"],
                "decision_rule": rule_label,
                "viral_threshold": result["viral_threshold"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "candidate_comparison_summary.csv", index=False)

    with open(
        OUTPUT_DIR / "final_test_evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "test_image_count": int(len(test_df)),
                "class_names": CLASS_NAMES,
                "preprocessing": {
                    "decoder": "tf.io.decode_image(channels=3, expand_animations=False)",
                    "resize": "tf.image.resize(224x224, bilinear)",
                    "dtype": "float32",
                    "pixel_range": "0..255",
                },
                "thresholds_selected_on_validation": {
                    "original_threshold_0_250": 0.250,
                    "source_balanced_threshold_0_275": 0.275,
                },
                "results": all_results,
                "warning": (
                    "Held-out test results are for final reporting only. "
                    "Do not tune thresholds or model settings using this test set."
                ),
            },
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("FINAL TEST COMPARISON")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("\nSaved all results to:")
    print(OUTPUT_DIR)
    print("\nIMPORTANT: Do not use these test results to tune thresholds or models.")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/55_final_test_classifier_evaluation.py