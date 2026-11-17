from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    CLASSIFICATION_DATA_DIR,
    SAVED_MODELS_DIR,
    IMAGE_SIZE,
)

BALANCED_DIR = ROOT / "data" / "processed" / "classification_source_balanced"
BALANCED_MANIFEST = BALANCED_DIR / "manifest.csv"
BALANCED_WEIGHTS = BALANCED_DIR / "class_weights.json"
ORIGINAL_MANIFEST = CLASSIFICATION_DATA_DIR / "manifest.csv"

INITIAL_CHECKPOINT = SAVED_MODELS_DIR / "efficientnetb0_pneumonia_classifier_best.keras"

OUTPUT_MODEL = SAVED_MODELS_DIR / "efficientnetb0_source_balanced_best.keras"
OUTPUT_LAST_MODEL = SAVED_MODELS_DIR / "efficientnetb0_source_balanced_last.keras"
OUTPUT_DIR = ROOT / "results" / "classification" / "source_balanced_training"
OUTPUT_HISTORY = OUTPUT_DIR / "training_history.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "training_summary.json"

BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
SEED = 42

REQUIRED_COLUMNS = ["split", "class_name", "processed_path"]


def resolve_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path

    candidates = [ROOT / path, CLASSIFICATION_DATA_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest(path, required_split):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[df["split"].astype(str).str.lower().eq(required_split)].copy()
    if df.empty:
        raise ValueError(f"No {required_split!r} rows found in {path}")

    df["label"] = df["class_name"].map(CLASS_TO_INDEX)
    if df["label"].isna().any():
        bad = sorted(df.loc[df["label"].isna(), "class_name"].astype(str).unique())
        raise ValueError(f"Unknown class names in {path}: {bad}")

    df["label"] = df["label"].astype("int32")
    df["absolute_path"] = df["processed_path"].map(resolve_path)

    missing_paths = [str(p) for p in df["absolute_path"] if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image files are missing. Examples:\n"
            + "\n".join(missing_paths[:10])
        )

    return df


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
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )
    image = tf.cast(image, tf.float32)  # Keep 0-255; EfficientNet has internal rescaling.
    label = tf.one_hot(label, depth=len(CLASS_NAMES))
    return image, label


def make_dataset(df, training):
    paths = df["absolute_path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    if not INITIAL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {INITIAL_CHECKPOINT}")

    for path in [OUTPUT_MODEL, OUTPUT_LAST_MODEL, OUTPUT_HISTORY, OUTPUT_SUMMARY]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}. "
                "Archive the previous experiment before rerunning."
            )

    if not BALANCED_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Balanced class weights not found: {BALANCED_WEIGHTS}. "
            "Run script 49 first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_manifest(BALANCED_MANIFEST, "train")
    # Use the existing, unchanged validation split from the original manifest.
    val_df = load_manifest(ORIGINAL_MANIFEST, "val")

    with BALANCED_WEIGHTS.open("r", encoding="utf-8") as file:
        weight_by_name = json.load(file)

    class_weights = {
        CLASS_TO_INDEX[name]: float(weight_by_name[name])
        for name in CLASS_NAMES
    }

    print("\nTRAINING DATA COUNTS")
    print(
        train_df.groupby(["source_dataset", "class_name"], dropna=False)
        .size()
        .rename("image_count")
        .reset_index()
        .to_string(index=False)
    )
    print(f"\nTraining images: {len(train_df):,}")
    print(f"Validation images: {len(val_df):,}")
    print(f"Class weights: {class_weights}")
    print("\nLoading original best checkpoint...")

    model = tf.keras.models.load_model(INITIAL_CHECKPOINT)

    # Controlled experiment: keep the EfficientNet backbone frozen and train
    # only the existing classification head, matching the prior class-weight
    # experiments. Do not alter the original checkpoint.
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = False

    # Keep all nested backbone layers frozen, including BatchNormalization.
    for layer in backbone.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trainable_params = int(
        sum(np.prod(variable.shape) for variable in model.trainable_variables)
    )
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {model.count_params():,}")

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_MODEL),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
    ]

    print("\nStarting source-balanced training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(OUTPUT_LAST_MODEL)
    pd.DataFrame(history.history).to_csv(OUTPUT_HISTORY, index=False)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    summary = {
        "experiment": "source_balanced_training_head_only",
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "best_checkpoint": str(OUTPUT_MODEL),
        "last_checkpoint": str(OUTPUT_LAST_MODEL),
        "training_manifest": str(BALANCED_MANIFEST),
        "validation_manifest": str(ORIGINAL_MANIFEST),
        "training_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "class_weights": {name: float(weight_by_name[name]) for name in CLASS_NAMES},
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "completed_epochs": int(len(history.history["loss"])),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch - 1]),
        "trainable_parameters": trainable_params,
        "test_split_accessed": False,
    }

    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, indent=4) + "\n",
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETE")
    print(json.dumps(summary, indent=4))
    print("\nThe original checkpoint was not modified.")
    print("The test split was not loaded or evaluated.")
    print("Use script 51 to compare this checkpoint on the same validation split.")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/50_train_source_balanced_efficientnet.py