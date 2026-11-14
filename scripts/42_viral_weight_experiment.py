import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    CLASSIFICATION_DATA_DIR, CLASSIFICATION_RESULTS_DIR, SAVED_MODELS_DIR,
    CLASS_NAMES, CLASS_TO_INDEX, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
)

SOURCE_MODEL_PATH = SAVED_MODELS_DIR / "efficientnetb0_pneumonia_classifier_best.keras"
MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"
CLASS_WEIGHTS_PATH = CLASSIFICATION_RESULTS_DIR / "pipeline" / "class_weights.json"
EXPERIMENT_ROOT = CLASSIFICATION_RESULTS_DIR / "viral_weight_experiments"

EXPERIMENTS = {"experiment_1_2x": 1.20, "experiment_1_4x": 1.40}
LEARNING_RATE = 1e-5
MAX_EPOCHS = 10
IMAGE_HEIGHT = IMAGE_WIDTH = IMAGE_SIZE
AUTOTUNE = tf.data.AUTOTUNE


def load_metadata():
    for path in (SOURCE_MODEL_PATH, MANIFEST_PATH, CLASS_WEIGHTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(MANIFEST_PATH)
    required = ["split", "class_name", "class_index", "processed_path"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    for split in ("train", "val"):
        if not (df["split"] == split).any():
            raise ValueError(f"No images found for split: {split}")

    for _, row in df.iterrows():
        expected = CLASS_TO_INDEX.get(row["class_name"])
        if expected is None or int(row["class_index"]) != expected:
            raise ValueError(f"Invalid class mapping in manifest: {row['class_name']}")

    with open(CLASS_WEIGHTS_PATH, encoding="utf-8") as file:
        saved = json.load(file)

    missing_weights = [name for name in CLASS_NAMES if name not in saved]
    if missing_weights:
        raise ValueError(f"Missing class weights: {missing_weights}")

    base_weights = {
        CLASS_TO_INDEX[name]: float(saved[name]) for name in CLASS_NAMES
    }
    print("Training images:", int((df["split"] == "train").sum()))
    print("Validation images:", int((df["split"] == "val").sum()))
    print("Test images will not be loaded or used.")
    return df, base_weights


def preprocess_image(image_path, label):
    image_data = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], method="bilinear")
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def build_dataset(df, split, training=False):
    part = df[df["split"] == split].copy()
    paths = part["processed_path"].astype(str).to_numpy()
    labels = part["class_index"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(part), 4096),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def run_experiment(df, base_weights, experiment_name, multiplier):
    print("\n" + "=" * 70)
    print(f"Starting {experiment_name}: Viral multiplier={multiplier}")
    print("=" * 70)

    experiment_dir = EXPERIMENT_ROOT / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_path = SAVED_MODELS_DIR / f"efficientnetb0_viral_weight_{multiplier:.1f}x_best.keras"
    history_csv = experiment_dir / "training_history.csv"
    history_json = experiment_dir / "training_history.json"
    weights_json = experiment_dir / "experiment_class_weights.json"
    summary_json = experiment_dir / "experiment_summary.json"

    for output in (model_path, history_csv, history_json, weights_json, summary_json):
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output}. "
                "Archive it or choose a new experiment name."
            )

    weights = dict(base_weights)
    viral_index = CLASS_TO_INDEX["Viral"]
    weights[viral_index] *= multiplier
    named_weights = {name: weights[CLASS_TO_INDEX[name]] for name in CLASS_NAMES}

    with open(weights_json, "w", encoding="utf-8") as file:
        json.dump(
            {"experiment": experiment_name, "viral_multiplier": multiplier,
             "class_weights": named_weights},
            file, indent=4,
        )

    print("Class weights:", named_weights)
    print("Loading the original best checkpoint afresh...")
    model = tf.keras.models.load_model(SOURCE_MODEL_PATH, compile=False)

    backbone = model.get_layer("efficientnetb0")
    if backbone.trainable:
        raise RuntimeError("Expected frozen EfficientNetB0 backbone.")
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print("Trainable parameters:", trainable_params)
    if trainable_params != 5124:
        raise RuntimeError(f"Expected 5124 trainable parameters, found {trainable_params}.")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    train_ds = build_dataset(df, "train", training=True)
    val_ds = build_dataset(df, "val", training=False)

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
        class_weight=weights, callbacks=callbacks, verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv, index=False)
    with open(history_json, "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=4)

    best_index = int(np.argmin(history.history["val_loss"]))
    summary = {
        "experiment": experiment_name,
        "source_checkpoint": str(SOURCE_MODEL_PATH),
        "viral_multiplier": multiplier,
        "viral_weight": named_weights["Viral"],
        "best_epoch": best_index + 1,
        "best_val_loss": float(history.history["val_loss"][best_index]),
        "val_accuracy_at_best_val_loss": float(history.history["val_accuracy"][best_index]),
        "checkpoint": str(model_path),
        "test_set_used": False,
    }
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("Experiment complete:", summary)
    del model, train_ds, val_ds
    tf.keras.backend.clear_session()
    return summary


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    print("TensorFlow:", tf.__version__)
    print("Source checkpoint:", SOURCE_MODEL_PATH)
    print("Learning rate:", LEARNING_RATE, "| Max epochs:", MAX_EPOCHS)

    df, base_weights = load_metadata()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name, multiplier in EXPERIMENTS.items():
        summaries.append(run_experiment(df, base_weights, name, multiplier))

    summary_path = EXPERIMENT_ROOT / "all_experiments_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)

    print("\nBoth experiments completed. Test set was not used.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
