
import sys
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
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
    SAVED_MODELS_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
)


ORIGINAL_MODEL_PATH = (
    SAVED_MODELS_DIR
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

FINETUNED_MODEL_PATH = (
    SAVED_MODELS_DIR
    / "efficientnetb0_pneumonia_classifier_finetuned_best.keras"
)

MANIFEST_PATH = (
    CLASSIFICATION_DATA_DIR / "manifest.csv"
)

OUTPUT_DIR = (
    CLASSIFICATION_RESULTS_DIR
    / "model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# VALIDATE FILES
# ============================================================

def validate_files():

    for path in [
        ORIGINAL_MODEL_PATH,
        FINETUNED_MODEL_PATH,
        MANIFEST_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(path)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "split",
        "class_name",
        "class_index",
        "processed_path",
    ]

    for column in required_columns:

        if column not in manifest.columns:
            raise ValueError(
                f"Missing manifest column: {column}"
            )

    validation_df = manifest[
        manifest["split"].astype(str).str.lower() == "val"
    ].copy()

    if validation_df.empty:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    validation_df = validation_df.reset_index(drop=True)

    paths = []

    for value in validation_df["processed_path"].astype(str):

        path = Path(value)

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(path)

        paths.append(str(path))

    labels = validation_df["class_index"].astype(
        np.int32
    ).to_numpy()

    return validation_df, paths, labels


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(path, label):

    image_data = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    image = tf.cast(image, tf.float32)

    return image, label


# ============================================================
# BUILD VALIDATION DATASET
# ============================================================

def build_validation_dataset(paths, labels):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_path, model_name, dataset, y_true):

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    probabilities = model.predict(
        dataset,
        verbose=1,
    )

    if probabilities.shape != (
        len(y_true),
        len(CLASS_NAMES),
    ):
        raise ValueError(
            f"Unexpected prediction shape: "
            f"{probabilities.shape}"
        )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    print("\nOverall metrics:")

    for key, value in metrics.items():

        if key != "model":
            print(f"{key}: {value:.6f}")

    print("\nClassification report:")
    print(report_text)

    print("\nConfusion matrix:")
    print(matrix)

    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix,
        "probabilities": probabilities,
        "predictions": y_pred,
    }


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

def save_model_results(
    model_name,
    result,
    validation_df,
    y_true,
):

    model_dir = OUTPUT_DIR / model_name

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df = pd.DataFrame(
        result["report"]
    ).transpose()

    report_df.to_csv(
        model_dir / "classification_report.csv"
    )

    matrix_df = pd.DataFrame(
        result["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_df.index.name = "actual_class"

    matrix_df.to_csv(
        model_dir / "confusion_matrix.csv"
    )

    prediction_df = validation_df.copy()

    prediction_df["actual_class_index"] = y_true

    prediction_df["predicted_class_index"] = (
        result["predictions"]
    )

    prediction_df["predicted_class"] = [
        CLASS_NAMES[index]
        for index in result["predictions"]
    ]

    prediction_df["correct"] = (
        prediction_df["actual_class_index"]
        == prediction_df["predicted_class_index"]
    )

    for index, class_name in enumerate(CLASS_NAMES):

        prediction_df[
            f"probability_{class_name.lower()}"
        ] = result["probabilities"][:, index]

    prediction_df.to_csv(
        model_dir / "validation_predictions.csv",
        index=False,
    )


# ============================================================
# COMPARE MODELS
# ============================================================

def save_comparison(original, finetuned):

    comparison_rows = []

    for model_name, result in [
        ("original", original),
        ("finetuned", finetuned),
    ]:

        comparison_rows.append(
            result["metrics"]
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "overall_model_comparison.csv",
        index=False,
    )

    class_rows = []

    for class_name in CLASS_NAMES:

        original_class = original["report"][class_name]
        finetuned_class = finetuned["report"][class_name]

        class_rows.append({
            "class": class_name,
            "original_precision": original_class["precision"],
            "finetuned_precision": finetuned_class["precision"],
            "original_recall": original_class["recall"],
            "finetuned_recall": finetuned_class["recall"],
            "original_f1": original_class["f1-score"],
            "finetuned_f1": finetuned_class["f1-score"],
            "original_support": original_class["support"],
            "finetuned_support": finetuned_class["support"],
        })

    class_comparison = pd.DataFrame(class_rows)

    class_comparison.to_csv(
        OUTPUT_DIR / "class_wise_comparison.csv",
        index=False,
    )

    return comparison_df, class_comparison


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ORIGINAL VS FINE-TUNED CLASSIFIER COMPARISON")
    print("=" * 70)

    validate_files()

    validation_df, paths, y_true = load_validation_data()

    print("\nValidation image count:", len(y_true))

    print("\nValidation class distribution:")

    print(
        validation_df["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES)
        .to_string()
    )

    dataset = build_validation_dataset(
        paths,
        y_true,
    )

    # --------------------------------------------------------
    # Evaluate original model
    # --------------------------------------------------------

    original = evaluate_model(
        ORIGINAL_MODEL_PATH,
        "original",
        dataset,
        y_true,
    )

    save_model_results(
        "original",
        original,
        validation_df,
        y_true,
    )

    # --------------------------------------------------------
    # Evaluate fine-tuned model
    # --------------------------------------------------------

    finetuned = evaluate_model(
        FINETUNED_MODEL_PATH,
        "finetuned",
        dataset,
        y_true,
    )

    save_model_results(
        "finetuned",
        finetuned,
        validation_df,
        y_true,
    )

    # --------------------------------------------------------
    # Overall and class-wise comparison
    # --------------------------------------------------------

    overall, class_wise = save_comparison(
        original,
        finetuned,
    )

    # --------------------------------------------------------
    # Viral -> Bacterial confusion
    # --------------------------------------------------------

    viral_index = CLASS_NAMES.index("Viral")
    bacterial_index = CLASS_NAMES.index("Bacterial")

    viral_actual = y_true == viral_index

    original_viral_to_bacterial = int(
        np.sum(
            (original["predictions"][viral_actual])
            == bacterial_index
        )
    )

    finetuned_viral_to_bacterial = int(
        np.sum(
            (finetuned["predictions"][viral_actual])
            == bacterial_index
        )
    )

    viral_confusion = pd.DataFrame([
        {
            "model": "original",
            "viral_to_bacterial": original_viral_to_bacterial,
        },
        {
            "model": "finetuned",
            "viral_to_bacterial": finetuned_viral_to_bacterial,
        },
    ])

    viral_confusion.to_csv(
        OUTPUT_DIR / "viral_to_bacterial_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL MODEL COMPARISON")
    print("=" * 70)

    print(overall.to_string(index=False))

    print("\nCLASS-WISE COMPARISON")

    print(class_wise.to_string(index=False))

    print("\nViral -> Bacterial errors:")

    print(viral_confusion.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- overall_model_comparison.csv")
    print("- class_wise_comparison.csv")
    print("- viral_to_bacterial_comparison.csv")
    print("- original/ classification_report.csv")
    print("- original/ confusion_matrix.csv")
    print("- original/ validation_predictions.csv")
    print("- finetuned/ classification_report.csv")
    print("- finetuned/ confusion_matrix.csv")
    print("- finetuned/ validation_predictions.csv")

    print("\nTest dataset was not used.")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/41_compare_classifier_models.py