import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
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
    / "pneumonia_classifier_ablation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = list(CLASS_NAMES)
NUM_CLASSES = len(CLASS_NAMES)

PROBABILITY_COLUMNS = [
    f"probability_{name}" for name in CLASS_NAMES
]


# ============================================================
# 2. LOAD AND VALIDATE EXISTING VALIDATION RECORDS
# ============================================================

print("\nLoading validation predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

required_columns = [
    "processed_path",
    "actual_class",
    "predicted_class",
    "source_dataset",
    *PROBABILITY_COLUMNS,
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if len(df) != 2962:
    raise ValueError(
        f"Expected 2962 validation images, found {len(df)}"
    )

if not set(df["actual_class"]).issubset(set(CLASS_NAMES)):
    raise ValueError("Unexpected actual class found.")

image_paths = df["processed_path"].tolist()

missing_images = [
    path for path in image_paths if not Path(path).is_file()
]

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} validation images are missing. "
        f"First missing image: {missing_images[0]}"
    )

y_true = np.array([
    CLASS_NAMES.index(label)
    for label in df["actual_class"]
])

saved_predictions = np.array([
    CLASS_NAMES.index(label)
    for label in df["predicted_class"]
])

saved_probabilities = df[PROBABILITY_COLUMNS].to_numpy(
    dtype=np.float32
)

print("Validation records:", len(df))
print("All validation images exist.")
print("Class order:", CLASS_NAMES)


# ============================================================
# 3. LOAD EXISTING BEST MODEL
# ============================================================

print("\nLoading pneumonia classifier...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

if model.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        "Model output count does not match the four classes."
    )


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def load_image(path):
    """
    Reproduce the original TensorFlow preprocessing:
    RGB decode, resize, float32, pixel range 0-255.
    """

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(image, tf.float32)

    return image


# ============================================================
# 5. MASKING FUNCTIONS
# ============================================================

def mask_border(image):
    """
    Replace the outer 15% border with image mean.
    """

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    border_y = tf.cast(
        tf.cast(height, tf.float32) * 0.15,
        tf.int32
    )

    border_x = tf.cast(
        tf.cast(width, tf.float32) * 0.15,
        tf.int32
    )

    rows = tf.range(height)[:, None]
    cols = tf.range(width)[None, :]

    keep_center = tf.logical_and(
        tf.logical_and(
            rows >= border_y,
            rows < height - border_y
        ),
        tf.logical_and(
            cols >= border_x,
            cols < width - border_x
        )
    )

    keep_center = keep_center[:, :, None]

    mean_value = tf.reduce_mean(
        image,
        axis=[0, 1],
        keepdims=True
    )

    return tf.where(
        keep_center,
        image,
        mean_value
    )


def mask_center(image):
    """
    Replace the central rectangular region with image mean.
    """

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    y_start = tf.cast(
        tf.cast(height, tf.float32) * 0.20,
        tf.int32
    )

    y_end = tf.cast(
        tf.cast(height, tf.float32) * 0.80,
        tf.int32
    )

    x_start = tf.cast(
        tf.cast(width, tf.float32) * 0.25,
        tf.int32
    )

    x_end = tf.cast(
        tf.cast(width, tf.float32) * 0.75,
        tf.int32
    )

    rows = tf.range(height)[:, None]
    cols = tf.range(width)[None, :]

    center_region = tf.logical_and(
        tf.logical_and(
            rows >= y_start,
            rows < y_end
        ),
        tf.logical_and(
            cols >= x_start,
            cols < x_end
        )
    )

    center_region = center_region[:, :, None]

    mean_value = tf.reduce_mean(
        image,
        axis=[0, 1],
        keepdims=True
    )

    return tf.where(
        center_region,
        mean_value,
        image
    )


# ============================================================
# 6. BUILD DATASET FOR EACH CONDITION
# ============================================================

def build_dataset(paths, condition="original"):

    dataset = tf.data.Dataset.from_tensor_slices(paths)

    def process(path):

        image = load_image(path)

        if condition == "border_masked":
            image = mask_border(image)

        elif condition == "central_masked":
            image = mask_center(image)

        return image

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# 7. RUN PREDICTIONS
# ============================================================

def predict_condition(condition):

    print(f"\nRunning condition: {condition}")

    dataset = build_dataset(
        image_paths,
        condition=condition
    )

    probabilities = model.predict(
        dataset,
        verbose=1
    )

    if probabilities.shape != (len(df), NUM_CLASSES):
        raise ValueError(
            f"Unexpected prediction shape: {probabilities.shape}"
        )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return probabilities, predictions


# ============================================================
# 8. ORIGINAL BASELINE VERIFICATION
# ============================================================

original_probabilities, original_predictions = (
    predict_condition("original")
)

original_agreement = np.mean(
    original_predictions == saved_predictions
)

original_probability_difference = np.mean(
    np.abs(original_probabilities - saved_probabilities)
)

print("\nOriginal baseline verification")
print("--------------------------------")
print("Prediction agreement:", original_agreement)
print(
    "Mean probability difference:",
    original_probability_difference
)

# Stop if the baseline does not reproduce the saved results.
if original_agreement != 1.0:
    raise RuntimeError(
        "Baseline predictions do not match saved predictions. "
        "Do not interpret masking results until this is investigated."
    )

if original_probability_difference > 0.0001:
    raise RuntimeError(
        "Baseline probabilities differ from saved predictions. "
        "Investigate preprocessing or model consistency."
    )

print("Baseline verification PASSED.")


# ============================================================
# 9. RUN MASKING EXPERIMENTS
# ============================================================

border_probabilities, border_predictions = (
    predict_condition("border_masked")
)

center_probabilities, center_predictions = (
    predict_condition("central_masked")
)


# ============================================================
# 10. METRICS
# ============================================================

def evaluate_condition(name, probabilities, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            labels=range(NUM_CLASSES),
            zero_division=0
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=range(NUM_CLASSES)
    )

    report = classification_report(
        y_true,
        predictions,
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    result = {
        "condition": name,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    print(f"\n{name.upper()} RESULTS")
    print("--------------------------------")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClass-wise report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=range(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(cm)

    return result, cm, report


# ============================================================
# 11. EVALUATE ALL CONDITIONS
# ============================================================

conditions = [
    ("original", original_probabilities, original_predictions),
    ("border_masked", border_probabilities, border_predictions),
    ("central_masked", center_probabilities, center_predictions),
]

summary_results = []
confusion_matrices = {}
class_reports = {}

for name, probabilities, predictions in conditions:

    result, cm, report = evaluate_condition(
        name,
        probabilities,
        predictions
    )

    summary_results.append(result)
    confusion_matrices[name] = cm.tolist()
    class_reports[name] = report


# ============================================================
# 12. SOURCE-WISE PERFORMANCE
# ============================================================

source_results = []

for condition_name, probabilities, predictions in conditions:

    for source in sorted(df["source_dataset"].dropna().unique()):

        mask = (
            df["source_dataset"].to_numpy() == source
        )

        source_accuracy = accuracy_score(
            y_true[mask],
            predictions[mask]
        )

        source_results.append({
            "condition": condition_name,
            "source_dataset": source,
            "sample_count": int(mask.sum()),
            "accuracy": float(source_accuracy),
        })

source_df = pd.DataFrame(source_results)

source_df.to_csv(
    OUTPUT_DIR / "source_wise_accuracy.csv",
    index=False
)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

prediction_frames = []

for name, probabilities, predictions in conditions:

    result_df = df[
        [
            "processed_path",
            "actual_class",
            "source_dataset",
        ]
    ].copy()

    result_df["condition"] = name

    result_df["predicted_class"] = [
        CLASS_NAMES[index] for index in predictions
    ]

    result_df["correct_prediction"] = (
        predictions == y_true
    )

    for index, class_name in enumerate(CLASS_NAMES):
        result_df[f"probability_{class_name}"] = (
            probabilities[:, index]
        )

    prediction_frames.append(result_df)

all_predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True
)

all_predictions_df.to_csv(
    OUTPUT_DIR / "ablation_predictions.csv",
    index=False
)


# ============================================================
# 14. SAVE SUMMARY AND REPORTS
# ============================================================

summary_df = pd.DataFrame(summary_results)

summary_df.to_csv(
    OUTPUT_DIR / "ablation_summary.csv",
    index=False
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

with open(
    OUTPUT_DIR / "classification_reports.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        class_reports,
        file,
        indent=4
    )

baseline_info = {
    "validation_samples": len(df),
    "class_names": CLASS_NAMES,
    "original_prediction_agreement": float(
        original_agreement
    ),
    "original_mean_probability_difference": float(
        original_probability_difference
    ),
    "baseline_verification": "PASSED",
}

with open(
    OUTPUT_DIR / "baseline_verification.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        baseline_info,
        file,
        indent=4
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("PNEUMONIA CLASSIFIER ABLATION COMPLETED")
print("========================================")

print(summary_df.to_string(index=False))

print("\nSource-wise accuracy:")
print(source_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR)


