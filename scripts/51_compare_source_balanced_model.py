from pathlib import Path
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
    SAVED_MODELS_DIR,
    IMAGE_SIZE,
)

BASELINE_CHECKPOINT = SAVED_MODELS_DIR / "efficientnetb0_pneumonia_classifier_best.keras"
BALANCED_CHECKPOINT = SAVED_MODELS_DIR / "efficientnetb0_source_balanced_best.keras"
VALIDATION_MANIFEST = CLASSIFICATION_DATA_DIR / "manifest.csv"
OUTPUT_DIR = CLASSIFICATION_RESULTS_DIR / "source_balanced_model_comparison"

BATCH_SIZE = 16
LABELS = list(range(len(CLASS_NAMES)))
PROBABILITY_COLUMNS = [f"probability_{name.lower()}" for name in CLASS_NAMES]


def resolve_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path

    candidates = [ROOT / path, CLASSIFICATION_DATA_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_validation_manifest():
    if not VALIDATION_MANIFEST.exists():
        raise FileNotFoundError(VALIDATION_MANIFEST)

    df = pd.read_csv(VALIDATION_MANIFEST)
    required = ["split", "class_name", "processed_path"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Validation manifest missing columns: {missing}")

    df = df[df["split"].astype(str).str.lower().eq("val")].copy()
    if df.empty:
        raise ValueError("No validation rows found.")

    df["label"] = df["class_name"].map(CLASS_TO_INDEX)
    if df["label"].isna().any():
        raise ValueError("Unknown class names in validation manifest.")

    df["label"] = df["label"].astype("int32")
    df["absolute_path"] = df["processed_path"].map(resolve_path)

    missing_paths = [str(path) for path in df["absolute_path"] if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} validation images are missing. "
            f"Examples: {missing_paths[:5]}"
        )

    return df


def decode_image(path, label):
    contents = tf.io.read_file(path)
    image = tf.io.decode_image(
        contents,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )
    image = tf.cast(image, tf.float32)  # 0-255, matching prior classifier pipeline.
    return image, label


def make_validation_dataset(df):
    dataset = tf.data.Dataset.from_tensor_slices((
        df["absolute_path"].astype(str).to_numpy(),
        df["label"].to_numpy(dtype=np.int32),
    ))
    dataset = dataset.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def predict_model(model_path, dataset, model_name):
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"\nLoading {model_name}: {model_path}")
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(dataset, verbose=1)
    del model
    return probabilities


def evaluate(name, y_true, probabilities, validation_df):
    if probabilities.shape != (len(y_true), len(CLASS_NAMES)):
        raise ValueError(
            f"{name}: unexpected prediction shape {probabilities.shape}"
        )

    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    overall = {
        "model": name,
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "viral_precision": float(precision[CLASS_TO_INDEX["Viral"]]),
        "viral_recall": float(recall[CLASS_TO_INDEX["Viral"]]),
        "viral_f1": float(f1[CLASS_TO_INDEX["Viral"]]),
        "validation_images": int(len(y_true)),
    }

    class_rows = []
    for index, class_name in enumerate(CLASS_NAMES):
        class_rows.append({
            "model": name,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "support": int(support[index]),
        })

    confusion_rows = []
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            confusion_rows.append({
                "model": name,
                "actual_class": actual_name,
                "predicted_class": predicted_name,
                "count": int(cm[actual_index, predicted_index]),
            })

    prediction_df = validation_df[
        ["processed_path", "class_name"]
    ].copy()
    prediction_df.insert(0, "model", name)
    prediction_df["actual_class_index"] = y_true
    prediction_df["predicted_class_index"] = y_pred
    prediction_df["predicted_class"] = [
        CLASS_NAMES[index] for index in y_pred
    ]
    for index, class_name in enumerate(CLASS_NAMES):
        prediction_df[f"probability_{class_name.lower()}"] = probabilities[:, index]

    return overall, class_rows, confusion_rows, prediction_df


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Archive the existing results before rerunning."
        )

    for checkpoint in [BASELINE_CHECKPOINT, BALANCED_CHECKPOINT]:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_df = load_validation_manifest()
    y_true = validation_df["label"].to_numpy(dtype=np.int32)
    validation_ds = make_validation_dataset(validation_df)

    print(f"\nValidation images: {len(validation_df):,}")
    print("Using the original, unchanged validation split.")
    print("No training or test images will be loaded.")

    model_specs = [
        ("original_baseline", BASELINE_CHECKPOINT),
        ("source_balanced", BALANCED_CHECKPOINT),
    ]

    overall_rows = []
    class_rows = []
    confusion_rows = []
    prediction_frames = []

    for model_name, checkpoint in model_specs:
        probabilities = predict_model(checkpoint, validation_ds, model_name)
        result = evaluate(model_name, y_true, probabilities, validation_df)
        overall, classes, confusions, predictions = result

        overall_rows.append(overall)
        class_rows.extend(classes)
        confusion_rows.extend(confusions)
        prediction_frames.append(predictions)

    overall_df = pd.DataFrame(overall_rows)
    class_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    overall_df.to_csv(OUTPUT_DIR / "overall_model_comparison.csv", index=False)
    class_df.to_csv(OUTPUT_DIR / "class_wise_comparison.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    print("\nOVERALL MODEL COMPARISON")
    print(overall_df.to_string(index=False))

    print("\nCLASS-WISE COMPARISON")
    print(class_df.to_string(index=False))

    print("\nCONFUSION MATRICES (rows=actual, columns=predicted)")
    for model_name in overall_df["model"]:
        subset = confusion_df[confusion_df["model"] == model_name]
        matrix = (
            subset.pivot(
                index="actual_class",
                columns="predicted_class",
                values="count",
            )
            .reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
            .fillna(0)
            .astype(int)
        )
        print(f"\n{model_name}")
        print(matrix.to_string())

    print(f"\nSaved comparison files to:\n{OUTPUT_DIR}")
    print("\nValidation split only. Test split was not accessed.")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/51_compare_source_balanced_model.py