import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow import keras

from aiml.config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SAVED_MODELS_DIR,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT /
    "data/processed/segmentation_covid/manifest.csv"
)

BASE_MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_balanced.keras"
)

BEST_MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_covid_finetuned_best.keras"
)

LAST_MODEL_PATH = (
    SAVED_MODELS_DIR /
    "unet_lung_segmentation_covid_finetuned_last.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results/segmentation/covid_domain_finetuning"
)

HISTORY_PATH = RESULTS_DIR / "finetuning_history.csv"
SUMMARY_PATH = RESULTS_DIR / "finetuning_summary.json"

# Conservative learning rate for fine-tuning.
FINE_TUNE_LEARNING_RATE = 0.00001

PATIENCE = 5


# ============================================================
# 2. LOAD DATASET PATHS
# ============================================================

def load_split_paths(split):

    df = pd.read_csv(MANIFEST_PATH)

    df = df[df["split"] == split].copy()

    if df.empty:
        raise ValueError(f"No records found for split: {split}")

    image_paths = df["processed_image_path"].astype(str).tolist()
    mask_paths = df["processed_mask_path"].astype(str).tolist()

    for image_path, mask_path in zip(image_paths, mask_paths):

        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)

        if not Path(mask_path).exists():
            raise FileNotFoundError(mask_path)

    print(f"{split} samples: {len(df)}")

    return image_paths, mask_paths


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def load_image_and_mask(image_path, mask_path):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image,
        channels=1,
    )

    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)

    mask = tf.io.decode_png(
        mask,
        channels=1,
    )

    mask = tf.cast(mask, tf.float32) / 255.0

    # Convert mask to binary values.
    mask = tf.cast(mask >= 0.5, tf.float32)

    tf.debugging.assert_equal(
        tf.shape(image),
        [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
    )

    tf.debugging.assert_equal(
        tf.shape(mask),
        [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
    )

    return image, mask


# ============================================================
# 4. BUILD TF.DATA PIPELINE
# ============================================================

def create_dataset(split, shuffle=False):

    image_paths, mask_paths = load_split_paths(split)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image_and_mask,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(image_paths)


# ============================================================
# 5. METRICS
# ============================================================

def dice_coefficient(y_true, y_pred):

    intersection = tf.reduce_sum(y_true * y_pred)

    denominator = (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred)
    )

    return (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )


def iou_score(y_true, y_pred):

    intersection = tf.reduce_sum(y_true * y_pred)

    union = (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) -
        intersection
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


# ============================================================
# 6. MAIN TRAINING
# ============================================================

def main():

    print("=" * 65)
    print("COVID-DOMAIN U-NET FINE-TUNING")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(BASE_MODEL_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training and validation datasets.
    train_dataset, train_count = create_dataset(
        "train",
        shuffle=True,
    )

    val_dataset, val_count = create_dataset(
        "val",
        shuffle=False,
    )

    print("\nDataset summary")
    print("-" * 45)
    print("Training images:", train_count)
    print("Validation images:", val_count)
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)
    print("Learning rate:", FINE_TUNE_LEARNING_RATE)

    # --------------------------------------------------------
    # LOAD EXISTING BALANCED U-NET
    # --------------------------------------------------------

    print("\nLoading original balanced U-Net...")

    model = keras.models.load_model(
        BASE_MODEL_PATH,
        compile=False,
    )

    print("Original model loaded successfully.")

    # --------------------------------------------------------
    # COMPILE MODEL FOR FINE-TUNING
    # --------------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            dice_coefficient,
            iou_score,
        ],
        run_eagerly=True,
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=str(LAST_MODEL_PATH),
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
            str(HISTORY_PATH)
        ),

        keras.callbacks.TerminateOnNaN(),
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting fine-tuning...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    best_epoch = (
        int(np.argmax(
            history.history["val_dice_coefficient"]
        )) + 1
    )

    best_val_dice = float(
        max(history.history["val_dice_coefficient"])
    )

    best_val_iou = float(
        max(history.history["val_iou_score"])
    )

    summary = {
        "experiment": "COVID-domain U-Net fine-tuning",
        "base_model": str(BASE_MODEL_PATH),
        "best_model": str(BEST_MODEL_PATH),
        "last_model": str(LAST_MODEL_PATH),
        "training_samples": train_count,
        "validation_samples": val_count,
        "learning_rate": FINE_TUNE_LEARNING_RATE,
        "maximum_epochs": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_validation_dice": best_val_dice,
        "best_validation_iou": best_val_iou,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("FINE-TUNING COMPLETED")
    print("=" * 65)

    print("Best epoch:", best_epoch)
    print("Best validation Dice:", round(best_val_dice, 4))
    print("Best validation IoU:", round(best_val_iou, 4))

    print("\nSaved files:")
    print("Best model:", BEST_MODEL_PATH)
    print("Last model:", LAST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()