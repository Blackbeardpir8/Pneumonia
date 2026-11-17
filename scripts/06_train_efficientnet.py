
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
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
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnetb0_pneumonia_classifier"

INITIAL_MODEL_PATH = (
    SAVED_MODELS_DIR / f"{MODEL_NAME}.keras"
)

BEST_MODEL_PATH = (
    SAVED_MODELS_DIR /
    f"{MODEL_NAME}_best.keras"
)

MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"

CLASS_WEIGHTS_PATH = (
    CLASSIFICATION_RESULTS_DIR /
    "pipeline" /
    "class_weights.json"
)

TRAINING_RESULTS_DIR = (
    CLASSIFICATION_RESULTS_DIR / "training"
)

IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE

MAX_EPOCHS = EPOCHS
INITIAL_LR = LEARNING_RATE

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# 3. LOAD MANIFEST AND CLASS WEIGHTS
# ============================================================

def load_training_metadata():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not INITIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Initial model not found: {INITIAL_MODEL_PATH}"
        )

    if not CLASS_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Class weights not found: {CLASS_WEIGHTS_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "split",
        "class_name",
        "class_index",
        "processed_path",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    # Check expected splits.
    for split in ["train", "val", "test"]:

        if not (df["split"] == split).any():
            raise ValueError(
                f"No images found for {split} split."
            )

    # Verify class names and indices.
    for _, row in df.iterrows():

        expected_index = CLASS_TO_INDEX.get(row["class_name"])

        if expected_index is None:
            raise ValueError(
                f"Unknown class: {row['class_name']}"
            )

        if int(row["class_index"]) != expected_index:
            raise ValueError(
                f"Incorrect class index: {row['class_name']}"
            )

    with open(
        CLASS_WEIGHTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        saved_weights = json.load(file)

    class_weights = {
        CLASS_TO_INDEX[class_name]: float(saved_weights[class_name])
        for class_name in CLASS_NAMES
    }

    print("\nTraining metadata loaded.")
    print("Total images:", len(df))

    print("\nClass weights:")
    for class_name in CLASS_NAMES:
        print(
            f"{class_name}: "
            f"{class_weights[CLASS_TO_INDEX[class_name]]:.4f}"
        )

    return df, class_weights


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path, label):

    image_data = tf.io.read_file(image_path)

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

    label = tf.cast(label, tf.int32)

    return image, label


# ============================================================
# 5. BUILD DATASETS
# ============================================================

def build_dataset(df, split, training=False):

    split_df = df[df["split"] == split].copy()

    image_paths = split_df["processed_path"].astype(
        str
    ).to_numpy()

    labels = split_df["class_index"].astype(
        np.int32
    ).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=min(len(split_df), 4096),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    print(
        f"{split.capitalize()} dataset:",
        len(split_df),
        "images"
    )

    return dataset


# ============================================================
# 6. LOAD AND COMPILE MODEL
# ============================================================

def load_and_compile_model():

    print("\nLoading initial EfficientNetB0 model...")

    model = tf.keras.models.load_model(
        INITIAL_MODEL_PATH,
        compile=False,
    )

    model.compile(
        optimizer=Adam(
            learning_rate=INITIAL_LR
        ),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("Model loaded and compiled successfully.")

    return model


# ============================================================
# 7. TRAINING CALLBACKS
# ============================================================

def create_callbacks():

    TRAINING_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BEST_MODEL_PATH.exists():
        raise FileExistsError(
            f"Best model already exists: {BEST_MODEL_PATH}. "
            "Archive it before starting another training run."
        )

    callbacks = [

        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks


# ============================================================
# 8. SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):

    history_df = pd.DataFrame(history.history)

    csv_path = (
        TRAINING_RESULTS_DIR / "training_history.csv"
    )

    history_df.to_csv(
        csv_path,
        index=False,
    )

    json_path = (
        TRAINING_RESULTS_DIR / "training_history.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history.history,
            file,
            indent=4,
        )

    print("\nTraining history saved:")
    print(csv_path)
    print(json_path)


# ============================================================
# 9. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("PNEUMONIA CDSS - EFFICIENTNETB0 INITIAL TRAINING")
    print("=" * 70)

    print("TensorFlow version:", tf.__version__)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", INITIAL_LR)

    # Step 1: Load metadata.
    df, class_weights = load_training_metadata()

    # Step 2: Build training and validation datasets.
    train_ds = build_dataset(
        df,
        "train",
        training=True,
    )

    val_ds = build_dataset(
        df,
        "val",
        training=False,
    )

    # The test dataset is deliberately not used during training.

    # Step 3: Load model.
    model = load_and_compile_model()

    print("\nTrainable parameters:")

    print(
        sum(
            int(np.prod(variable.shape))
            for variable in model.trainable_variables
        )
    )

    # Step 4: Prepare callbacks.
    callbacks = create_callbacks()

    # Step 5: Start training.
    print("\nStarting model training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 6: Save history.
    save_training_history(history)

    print("\nBest model saved at:")
    print(BEST_MODEL_PATH)

    print("\n" + "=" * 70)
    print("INITIAL EFFICIENTNETB0 TRAINING COMPLETED")
    print("The test dataset has not been used.")
    print("=" * 70)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/06_train_efficientnet.py