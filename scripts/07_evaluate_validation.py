
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
    SAVED_MODELS_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_PATH = (
    SAVED_MODELS_DIR /
    "efficientnetb0_pneumonia_classifier_best.keras"
)

MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"

EVALUATION_DIR = (
    CLASSIFICATION_RESULTS_DIR / "validation_evaluation"
)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 3. LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found: {MODEL_PATH}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    val_df = df[df["split"] == "val"].copy()

    if val_df.empty:
        raise ValueError("Validation dataset is empty.")

    image_paths = val_df["processed_path"].astype(
        str
    ).to_numpy()

    labels = val_df["class_index"].astype(
        np.int32
    ).to_numpy()

    missing_paths = [
        path for path in image_paths
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} validation images are missing."
        )

    print("Validation images:", len(val_df))

    return val_df, image_paths, labels


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image_data = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )

    return tf.cast(image, tf.float32)


# ============================================================
# 5. BUILD VALIDATION DATASET
# ============================================================

def build_validation_dataset(image_paths):

    dataset = tf.data.Dataset.from_tensor_slices(
        image_paths
    )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# 6. LOAD MODEL AND GENERATE PREDICTIONS
# ============================================================

def generate_predictions(dataset):

    print("\nLoading best EfficientNetB0 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    print("\nGenerating validation predictions...")

    probabilities = model.predict(
        dataset,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    if probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            "Model output does not match the number of classes."
        )

    return probabilities, predictions


# ============================================================
# 7. EVALUATE CLASSIFICATION PERFORMANCE
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    probabilities,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    print("\nVALIDATION ACCURACY")
    print("-" * 45)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")
    print("-" * 45)

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    print(matrix_df)

    return accuracy, matrix_df, report_df


# ============================================================
# 8. SAVE EVALUATION RESULTS
# ============================================================

def save_results(
    val_df,
    y_true,
    y_pred,
    probabilities,
    accuracy,
    matrix_df,
    report_df,
):

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save confusion matrix.
    matrix_df.to_csv(
        EVALUATION_DIR / "confusion_matrix.csv"
    )

    # Save classification report.
    report_df.to_csv(
        EVALUATION_DIR / "classification_report.csv"
    )

    # Save individual predictions.
    predictions_df = val_df[
        [
            "processed_path",
            "class_name",
            "class_index",
            "source_dataset",
            "source_class_id",
            "sha256",
        ]
    ].copy()

    predictions_df["actual_class"] = [
        CLASS_NAMES[index] for index in y_true
    ]

    predictions_df["predicted_class"] = [
        CLASS_NAMES[index] for index in y_pred
    ]

    predictions_df["correct_prediction"] = (
        y_true == y_pred
    )

    for index, class_name in enumerate(CLASS_NAMES):

        predictions_df[
            f"probability_{class_name}"
        ] = probabilities[:, index]

    predictions_df.to_csv(
        EVALUATION_DIR / "validation_predictions.csv",
        index=False,
    )

    # Save summary.
    summary = {
        "model": "EfficientNetB0",
        "evaluation_split": "validation",
        "total_images": int(len(y_true)),
        "accuracy": float(accuracy),
        "accuracy_percentage": float(accuracy * 100),
        "class_names": CLASS_NAMES,
        "confusion_matrix": matrix_df.values.tolist(),
    }

    with open(
        EVALUATION_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nEvaluation reports saved:")
    print(EVALUATION_DIR)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PNEUMONIA CDSS - VALIDATION EVALUATION")
    print("=" * 65)

    val_df, image_paths, y_true = load_validation_data()

    val_ds = build_validation_dataset(image_paths)

    probabilities, y_pred = generate_predictions(val_ds)

    accuracy, matrix_df, report_df = evaluate_model(
        y_true,
        y_pred,
        probabilities,
    )

    save_results(
        val_df,
        y_true,
        y_pred,
        probabilities,
        accuracy,
        matrix_df,
        report_df,
    )

    print("\n" + "=" * 65)
    print("VALIDATION EVALUATION COMPLETED")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
