
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
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import IMAGE_SIZE, BATCH_SIZE


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_classifier"
)

MODEL_DIR = PROJECT_ROOT / "saved_models"

MODEL_PATH = MODEL_DIR / "dataset_source_classifier.keras"

RANDOM_SEED = 42
EPOCHS = 15
LEARNING_RATE = 0.001

SELECTED_CLASSES = ["Normal", "Viral"]

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# LOAD MANIFEST
# --------------------------------------------------

def load_manifest():

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    df = pd.read_csv(MANIFEST_FILE)

    required_columns = [
        "split",
        "class_name",
        "source_dataset",
        "processed_path",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing manifest columns: {missing}"
        )

    df = df[
        df["class_name"].isin(SELECTED_CLASSES)
        & df["split"].isin(["train", "val"])
    ].copy()

    if df.empty:
        raise ValueError(
            "No Normal/Viral training or validation records found."
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
            f"{len(missing_files)} manifest image paths do not exist. "
            "Check processed_path values."
        )

    sources = sorted(df["source_dataset"].unique())

    if len(sources) != 2:
        raise ValueError(
            f"Expected exactly two dataset sources, found: {sources}"
        )

    print("\nAvailable source datasets:")
    for source in sources:
        print(f"  {source}")

    return df, sources


# --------------------------------------------------
# BALANCE SOURCE AND CLASS COUNTS
# --------------------------------------------------

def balance_split(df, split_name):

    split_df = df[
        df["split"] == split_name
    ].copy()

    counts = (
        split_df.groupby(
            ["source_dataset", "class_name"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(f"\n{split_name.upper()} counts before balancing:")
    print(counts.to_string())

    groups = []

    for source in sorted(split_df["source_dataset"].unique()):

        for class_name in SELECTED_CLASSES:

            group = split_df[
                (split_df["source_dataset"] == source)
                & (split_df["class_name"] == class_name)
            ]

            if group.empty:
                raise ValueError(
                    f"No {class_name} images for source {source} "
                    f"in {split_name} split."
                )

            groups.append(group)

    balanced_count = min(len(group) for group in groups)

    if balanced_count < 1:
        raise ValueError(
            f"Not enough samples to balance {split_name}."
        )

    balanced_groups = []

    for group in groups:

        selected = group.sample(
            n=balanced_count,
            random_state=RANDOM_SEED
        )

        balanced_groups.append(selected)

    balanced_df = pd.concat(
        balanced_groups,
        ignore_index=True
    )

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(
        f"\n{split_name.upper()} balanced samples "
        f"per source/class group: {balanced_count}"
    )

    print(
        balanced_df.groupby(
            ["source_dataset", "class_name"]
        ).size().unstack(fill_value=0).to_string()
    )

    return balanced_df


# --------------------------------------------------
# SOURCE LABELS
# --------------------------------------------------

def assign_source_labels(df, sources):

    source_to_index = {
        source: index
        for index, source in enumerate(sources)
    }

    df = df.copy()

    df["source_label"] = df["source_dataset"].map(
        source_to_index
    )

    return df


# --------------------------------------------------
# TF.DATA PIPELINE
# --------------------------------------------------

def load_image(image_path, label):

    image_data = tf.io.read_file(image_path)

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

    image = tf.cast(image, tf.float32)

    return image, label


def build_dataset(df, training=False):

    paths = df["processed_path"].values.astype(str)

    labels = df["source_label"].values.astype(
        np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# --------------------------------------------------
# BUILD SOURCE CLASSIFIER
# --------------------------------------------------

def build_model():

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    x = tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu"
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="dataset_source_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


# --------------------------------------------------
# EVALUATE SOURCE CLASSIFIER
# --------------------------------------------------

def evaluate_model(model, validation_dataset, val_df, sources):

    probabilities = model.predict(
        validation_dataset,
        verbose=1
    ).flatten()

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    actual_labels = val_df["source_label"].values

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    auc = roc_auc_score(
        actual_labels,
        probabilities
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=sources,
        output_dict=True,
        zero_division=0
    )

    print("\nSOURCE CLASSIFIER RESULTS")
    print("=" * 50)

    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Validation ROC-AUC: {auc:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            actual_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=sources,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(matrix)

    # Save validation predictions

    predictions_df = val_df[
        [
            "processed_path",
            "source_dataset",
            "class_name"
        ]
    ].copy()

    predictions_df["actual_source"] = [
        sources[label]
        for label in actual_labels
    ]

    predictions_df["predicted_source"] = [
        sources[label]
        for label in predicted_labels
    ]

    predictions_df["source_probability"] = probabilities

    predictions_df["correct_prediction"] = (
        actual_labels == predicted_labels
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "source_predictions.csv",
        index=False
    )

    # Save confusion matrix

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{source}"
            for source in sources
        ],
        columns=[
            f"Predicted_{source}"
            for source in sources
        ]
    )

    matrix_df.to_csv(
        OUTPUT_DIR / "source_confusion_matrix.csv"
    )

    # Save classification report

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        OUTPUT_DIR / "source_classification_report.csv"
    )

    summary = {
        "validation_images": int(len(val_df)),
        "validation_accuracy": float(accuracy),
        "validation_accuracy_percent": float(accuracy * 100),
        "validation_roc_auc": float(auc),
        "sources": sources,
        "selected_classes": SELECTED_CLASSES,
        "confusion_matrix": matrix.tolist(),
    }

    with open(
        OUTPUT_DIR / "source_classifier_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    return summary


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading classification manifest...")

    df, sources = load_manifest()

    # Balance training and validation independently.
    train_df = balance_split(df, "train")
    val_df = balance_split(df, "val")

    train_df = assign_source_labels(
        train_df,
        sources
    )

    val_df = assign_source_labels(
        val_df,
        sources
    )

    print("\nFinal dataset sizes:")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    train_dataset = build_dataset(
        train_df,
        training=True
    )

    val_dataset = build_dataset(
        val_df,
        training=False
    )

    print("\nBuilding source classifier...")

    model = build_model()

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\nStarting source classifier training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print("\nLoading best source classifier checkpoint...")

    best_model = tf.keras.models.load_model(
        MODEL_PATH
    )

    summary = evaluate_model(
        best_model,
        val_dataset,
        val_df,
        sources
    )

    print("\n" + "=" * 55)
    print("DATASET SOURCE EXPERIMENT COMPLETED")
    print("=" * 55)

    print(
        f"Validation accuracy: "
        f"{summary['validation_accuracy_percent']:.2f}%"
    )

    print(
        f"Validation ROC-AUC: "
        f"{summary['validation_roc_auc']:.4f}"
    )

    print(f"Model saved: {MODEL_PATH}")
    print(f"Reports saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/11_test_dataset_source_bias.py