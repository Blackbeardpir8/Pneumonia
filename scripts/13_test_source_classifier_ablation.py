
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import IMAGE_SIZE, BATCH_SIZE


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "dataset_source_classifier.keras"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_classifier"
    / "source_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_classifier"
    / "ablation_corrected"
)

SOURCE_NAMES = [
    "bacterial_viral_dataset",
    "covid19_radiography",
]

SOURCE_TO_INDEX = {
    source: index
    for index, source in enumerate(SOURCE_NAMES)
}

RANDOM_SEED = 42

BORDER_FRACTION = 0.15

CENTRAL_X_START = 0.25
CENTRAL_X_END = 0.75
CENTRAL_Y_START = 0.20
CENTRAL_Y_END = 0.80

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# LOAD MODEL AND VALIDATION RECORDS
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Source classifier loaded.")

    return model


def load_predictions():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "processed_path",
        "actual_source",
        "source_probability",
        "predicted_source",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.copy()

    df = df[
        df["actual_source"].isin(SOURCE_NAMES)
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No validation records found."
        )

    df["processed_path"] = df["processed_path"].apply(
        lambda value: str(
            Path(value)
            if Path(value).is_absolute()
            else PROJECT_ROOT / value
        )
    )

    missing_files = df[
        ~df["processed_path"].apply(
            lambda value: Path(value).is_file()
        )
    ]

    if not missing_files.empty:
        raise FileNotFoundError(
            f"{len(missing_files)} image files are missing."
        )

    df["actual_label"] = df["actual_source"].map(
        SOURCE_TO_INDEX
    )

    df["saved_probability"] = pd.to_numeric(
        df["source_probability"],
        errors="raise"
    )

    df["saved_prediction"] = df["predicted_source"].map(
        SOURCE_TO_INDEX
    )

    if df["saved_prediction"].isnull().any():
        raise ValueError(
            "Unknown predicted source in saved predictions."
        )

    return df


# --------------------------------------------------
# EXACT TENSORFLOW IMAGE PREPROCESSING
# --------------------------------------------------

def load_image_tensor(image_path):

    image_data = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


def load_all_images(df):

    print("\nLoading images with TensorFlow preprocessing...")

    images = []

    for index, path in enumerate(
        df["processed_path"].tolist()
    ):

        image = load_image_tensor(path)

        images.append(
            image.numpy()
        )

        if (index + 1) % 100 == 0:
            print(
                f"Loaded {index + 1}/{len(df)} images"
            )

    images = np.asarray(
        images,
        dtype=np.float32
    )

    print("Image tensor shape:", images.shape)
    print("Image dtype:", images.dtype)
    print(
        "Pixel range:",
        float(images.min()),
        float(images.max())
    )

    return images


# --------------------------------------------------
# MASKING FUNCTIONS
# --------------------------------------------------

def apply_original(images):

    return images.copy()


def apply_border_mask(images):

    masked = images.copy()

    height, width = masked.shape[1:3]

    x_margin = int(
        width * BORDER_FRACTION
    )

    y_margin = int(
        height * BORDER_FRACTION
    )

    # Use each image's own mean pixel value.
    fill_values = np.mean(
        images,
        axis=(1, 2),
        keepdims=True
    )

    masked[:] = fill_values

    masked[
        :,
        y_margin:height - y_margin,
        x_margin:width - x_margin,
        :
    ] = images[
        :,
        y_margin:height - y_margin,
        x_margin:width - x_margin,
        :
    ]

    return masked


def apply_central_mask(images):

    masked = images.copy()

    height, width = images.shape[1:3]

    x_start = int(
        width * CENTRAL_X_START
    )

    x_end = int(
        width * CENTRAL_X_END
    )

    y_start = int(
        height * CENTRAL_Y_START
    )

    y_end = int(
        height * CENTRAL_Y_END
    )

    fill_values = np.mean(
        images,
        axis=(1, 2),
        keepdims=True
    )

    masked[
        :,
        y_start:y_end,
        x_start:x_end,
        :
    ] = fill_values

    return masked


# --------------------------------------------------
# BASELINE CHECK
# --------------------------------------------------

def verify_baseline(
    model,
    images,
    df
):

    print("\nVerifying original-image baseline...")

    probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    labels = df["actual_label"].to_numpy()

    saved_probabilities = (
        df["saved_probability"].to_numpy()
    )

    saved_predictions = (
        df["saved_prediction"].to_numpy()
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    auc = roc_auc_score(
        labels,
        probabilities
    )

    agreement = np.mean(
        predictions == saved_predictions
    )

    probability_difference = np.mean(
        np.abs(
            probabilities - saved_probabilities
        )
    )

    print("\nBASELINE VERIFICATION")
    print("=" * 50)

    print(
        f"Current accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Current ROC-AUC: {auc:.4f}"
    )

    print(
        f"Prediction agreement: {agreement * 100:.2f}%"
    )

    print(
        f"Mean probability difference: "
        f"{probability_difference:.10f}"
    )

    # Do not continue if preprocessing does not
    # reproduce the saved source-classifier predictions.

    if agreement < 0.999:
        raise RuntimeError(
            "Baseline prediction agreement is below 99.9%. "
            "Stop the ablation experiment and investigate preprocessing."
        )

    if probability_difference > 1e-5:
        raise RuntimeError(
            "Baseline probability difference is too high. "
            "Stop and investigate preprocessing."
        )

    print("\nBaseline verification PASSED.")

    return probabilities


# --------------------------------------------------
# EVALUATE ONE EXPERIMENT
# --------------------------------------------------

def evaluate_experiment(
    model,
    images,
    df,
    experiment_name
):

    print(
        f"\nRunning experiment: {experiment_name}"
    )

    probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    ).flatten()

    labels = df["actual_label"].to_numpy()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    auc = roc_auc_score(
        labels,
        probabilities
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=SOURCE_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "-" * 55)
    print(f"EXPERIMENT: {experiment_name}")
    print("-" * 55)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    # Save per-image predictions.

    predictions_df = df[
        [
            "processed_path",
            "actual_source",
        ]
    ].copy()

    predictions_df["predicted_source"] = [
        SOURCE_NAMES[value]
        for value in predictions
    ]

    predictions_df["source_probability"] = probabilities

    predictions_df["correct_prediction"] = (
        predictions == labels
    )

    predictions_df["experiment"] = experiment_name

    predictions_df.to_csv(
        OUTPUT_DIR
        / f"{experiment_name}_predictions.csv",
        index=False
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{source}"
            for source in SOURCE_NAMES
        ],
        columns=[
            f"Predicted_{source}"
            for source in SOURCE_NAMES
        ]
    )

    matrix_df.to_csv(
        OUTPUT_DIR
        / f"{experiment_name}_confusion_matrix.csv"
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        OUTPUT_DIR
        / f"{experiment_name}_classification_report.csv"
    )

    return {
        "experiment": experiment_name,
        "total_images": int(len(labels)),
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "roc_auc": float(auc),
        "correct_predictions": int(
            np.sum(predictions == labels)
        ),
        "incorrect_predictions": int(
            np.sum(predictions != labels)
        ),
        "confusion_matrix": matrix.tolist(),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading source classifier...")

    model = load_model()

    print("\nLoading validation records...")

    df = load_predictions()

    print(
        f"Total validation images: {len(df)}"
    )

    images = load_all_images(df)

    # First, establish reproducible baseline.

    baseline_probabilities = verify_baseline(
        model,
        images,
        df
    )

    results = []

    # Original condition reuses verified probabilities.

    original_predictions = (
        baseline_probabilities >= 0.5
    ).astype(int)

    labels = df["actual_label"].to_numpy()

    original_accuracy = accuracy_score(
        labels,
        original_predictions
    )

    original_auc = roc_auc_score(
        labels,
        baseline_probabilities
    )

    original_matrix = confusion_matrix(
        labels,
        original_predictions,
        labels=[0, 1]
    )

    results.append({
        "experiment": "original",
        "total_images": int(len(labels)),
        "accuracy": float(original_accuracy),
        "accuracy_percent": float(original_accuracy * 100),
        "roc_auc": float(original_auc),
        "correct_predictions": int(
            np.sum(original_predictions == labels)
        ),
        "incorrect_predictions": int(
            np.sum(original_predictions != labels)
        ),
        "confusion_matrix": original_matrix.tolist(),
    })

    # Save verified original predictions.

    original_df = df[
        ["processed_path", "actual_source"]
    ].copy()

    original_df["predicted_source"] = [
        SOURCE_NAMES[value]
        for value in original_predictions
    ]

    original_df["source_probability"] = baseline_probabilities

    original_df["correct_prediction"] = (
        original_predictions == labels
    )

    original_df["experiment"] = "original"

    original_df.to_csv(
        OUTPUT_DIR / "original_predictions.csv",
        index=False
    )

    # Run corrected masking experiments.

    experiments = {
        "border_masked": apply_border_mask,
        "central_masked": apply_central_mask,
    }

    for experiment_name, mask_function in experiments.items():

        modified_images = mask_function(images)

        result = evaluate_experiment(
            model,
            modified_images,
            df,
            experiment_name
        )

        results.append(result)

        del modified_images

    # Save summary.

    summary_df = pd.DataFrame([
        {
            key: value
            for key, value in result.items()
            if key != "confusion_matrix"
        }
        for result in results
    ])

    summary_df.to_csv(
        OUTPUT_DIR / "ablation_comparison.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "ablation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("CORRECTED ABLATION EXPERIMENT COMPLETED")
    print("=" * 60)

    print(
        summary_df.to_string(index=False)
    )

    print(
        f"\nReports saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()


# to run script, use the following command:
# python scripts/13_test_source_classifier_ablation.py