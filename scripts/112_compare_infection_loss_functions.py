
import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
)

EXPERIMENTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "loss_function_experiments"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "saved_models"
    / "infection_loss_experiments"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
MASK_THRESHOLD = 0.5

POSITIVE_PIXEL_WEIGHT = 2.0

AUTOTUNE = tf.data.AUTOTUNE

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def reset_random_seeds():

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 3. DATASET PATH COLLECTION
# ============================================================

def collect_image_mask_pairs(split):

    split_dir = DATASET_ROOT / split / "COVID-19"

    image_dir = split_dir / "images"
    mask_dir = split_dir / "infection masks"

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    missing_masks = set(images) - set(masks)
    extra_masks = set(masks) - set(images)

    if missing_masks or extra_masks:
        raise ValueError(
            f"{split}: image-mask mismatch. "
            f"Missing masks={len(missing_masks)}, "
            f"extra masks={len(extra_masks)}"
        )

    if not images:
        raise ValueError(f"No images found for {split}")

    pairs = [
        (str(images[name]), str(masks[name]))
        for name in sorted(images)
    ]

    print(f"{split}: {len(pairs)} matched pairs")

    return pairs


# ============================================================
# 4. IMAGE AND MASK PREPROCESSING
# ============================================================

def load_image_and_mask(image_path, mask_path):

    image_bytes = tf.io.read_file(image_path)
    mask_bytes = tf.io.read_file(mask_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    mask = tf.io.decode_image(
        mask_bytes,
        channels=1,
        expand_animations=False,
    )

    image.set_shape([None, None, 1])
    mask.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = tf.cast(mask >= 128.0, tf.float32)

    return image, mask


# ============================================================
# 5. BUILD DATASET
# ============================================================

def create_dataset(pairs, training=False):

    image_paths = [pair[0] for pair in pairs]
    mask_paths = [pair[1] for pair in pairs]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(pairs),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image_and_mask,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# 6. LIGHTWEIGHT U-NET ARCHITECTURE
# ============================================================

def convolution_block(inputs, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def build_infection_unet():

    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    )

    enc1 = convolution_block(inputs, 16)
    pool1 = layers.MaxPooling2D(2)(enc1)

    enc2 = convolution_block(pool1, 32)
    pool2 = layers.MaxPooling2D(2)(enc2)

    enc3 = convolution_block(pool2, 64)
    pool3 = layers.MaxPooling2D(2)(enc3)

    bottleneck = convolution_block(pool3, 128)

    up3 = layers.UpSampling2D(
        2,
        interpolation="bilinear",
    )(bottleneck)

    up3 = layers.Concatenate()([up3, enc3])
    dec3 = convolution_block(up3, 64)

    up2 = layers.UpSampling2D(
        2,
        interpolation="bilinear",
    )(dec3)

    up2 = layers.Concatenate()([up2, enc2])
    dec2 = convolution_block(up2, 32)

    up1 = layers.UpSampling2D(
        2,
        interpolation="bilinear",
    )(dec2)

    up1 = layers.Concatenate()([up1, enc1])
    dec1 = convolution_block(up1, 16)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="infection_probability_mask",
    )(dec1)

    return keras.Model(
        inputs,
        outputs,
        name="covid_infection_unet",
    )


# ============================================================
# 7. LOSS FUNCTIONS
# ============================================================

def dice_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3],
    )

    denominator = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    return 1.0 - tf.reduce_mean(dice)


def baseline_bce_dice_loss(y_true, y_pred):

    bce = keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )

    bce = tf.reduce_mean(bce)

    return bce + dice_loss(y_true, y_pred)


def weighted_bce_dice_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate pixel-wise binary cross-entropy.
    bce = keras.backend.binary_crossentropy(
        y_true,
        y_pred,
    )

    # Give positive infection pixels a higher weight.
    pixel_weights = (
        1.0
        + (POSITIVE_PIXEL_WEIGHT - 1.0) * y_true
    )

    weighted_bce = tf.reduce_mean(
        bce * pixel_weights
    )

    return weighted_bce + dice_loss(y_true, y_pred)


def tversky_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    alpha = 0.3
    beta = 0.7

    true_positive = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3],
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true) * y_pred,
        axis=[1, 2, 3],
    )

    false_negative = tf.reduce_sum(
        y_true * (1.0 - y_pred),
        axis=[1, 2, 3],
    )

    tversky_index = (
        true_positive + 1e-6
    ) / (
        true_positive
        + alpha * false_positive
        + beta * false_negative
        + 1e-6
    )

    return 1.0 - tf.reduce_mean(tversky_index)


# ============================================================
# 8. METRICS
# ============================================================

def dice_coefficient(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(
        y_pred >= MASK_THRESHOLD,
        tf.float32,
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3],
    )

    denominator = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    return tf.reduce_mean(dice)


def iou_score(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(
        y_pred >= MASK_THRESHOLD,
        tf.float32,
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3],
    )

    union = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - intersection
    )

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return tf.reduce_mean(iou)


# ============================================================
# 9. SELECT EXPERIMENT
# ============================================================

def get_experiment_config(experiment):

    if experiment == "baseline":
        return {
            "loss": baseline_bce_dice_loss,
            "description": "Original BCE + Dice loss",
        }

    if experiment == "weighted_bce_dice":
        return {
            "loss": weighted_bce_dice_loss,
            "description": (
                "Weighted BCE + Dice; "
                f"positive pixel weight={POSITIVE_PIXEL_WEIGHT}"
            ),
        }

    if experiment == "tversky":
        return {
            "loss": tversky_loss,
            "description": (
                "Tversky loss; alpha=0.3, beta=0.7"
            ),
        }

    raise ValueError(f"Unknown experiment: {experiment}")


# ============================================================
# 10. TRAIN ONE EXPERIMENT
# ============================================================

def main(experiment):

    reset_random_seeds()
    keras.backend.clear_session()

    config = get_experiment_config(experiment)

    experiment_results_dir = EXPERIMENTS_DIR / experiment
    experiment_models_dir = MODELS_DIR / experiment

    experiment_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        experiment_models_dir / "best_model.keras"
    )

    last_model_path = (
        experiment_models_dir / "last_model.keras"
    )

    history_path = (
        experiment_results_dir / "training_history.csv"
    )

    summary_path = (
        experiment_results_dir / "summary.json"
    )

    print("=" * 70)
    print("COVID INFECTION SEGMENTATION LOSS EXPERIMENT")
    print("=" * 70)

    print("Experiment:", experiment)
    print("Description:", config["description"])
    print("Seed:", SEED)
    print("Image size:", IMAGE_HEIGHT, "x", IMAGE_WIDTH)
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Metric threshold:", MASK_THRESHOLD)

    if best_model_path.exists():
        raise FileExistsError(
            f"Experiment checkpoint already exists: {best_model_path}\n"
            "Do not overwrite an earlier experiment. "
            "Rename or back up its output folder first."
        )

    if history_path.exists():
        raise FileExistsError(
            f"Experiment history already exists: {history_path}\n"
            "Do not overwrite an earlier experiment."
        )

    train_pairs = collect_image_mask_pairs("Train")
    val_pairs = collect_image_mask_pairs("Val")

    print("\nLoading training dataset...")
    train_dataset = create_dataset(
        train_pairs,
        training=True,
    )

    print("Loading validation dataset...")
    val_dataset = create_dataset(
        val_pairs,
        training=False,
    )

    print("\nBuilding a fresh U-Net...")
    model = build_infection_unet()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss=config["loss"],
        metrics=[
            dice_coefficient,
            iou_score,
            keras.metrics.Precision(
                thresholds=MASK_THRESHOLD,
                name="precision",
            ),
            keras.metrics.Recall(
                thresholds=MASK_THRESHOLD,
                name="recall",
            ),
        ],
    )

    model.summary()

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
            str(history_path),
        ),

        keras.callbacks.TerminateOnNaN(),
    ]

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(last_model_path)

    best_epoch = (
        int(
            np.argmax(
                history.history["val_dice_coefficient"]
            )
        ) + 1
    )

    summary = {
        "experiment": experiment,
        "description": config["description"],
        "seed": SEED,
        "train_samples": len(train_pairs),
        "validation_samples": len(val_pairs),
        "epochs_completed": len(
            history.history["loss"]
        ),
        "best_epoch": best_epoch,
        "best_validation_dice": float(
            max(history.history["val_dice_coefficient"])
        ),
        "best_validation_iou": float(
            max(history.history["val_iou_score"])
        ),
        "best_validation_precision": float(
            history.history["val_precision"][
                best_epoch - 1
            ]
        ),
        "best_validation_recall": float(
            history.history["val_recall"][
                best_epoch - 1
            ]
        ),
        "threshold": MASK_THRESHOLD,
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "history_path": str(history_path),
        "test_set_used": False,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print("Experiment:", experiment)
    print("Best epoch:", best_epoch)
    print(
        "Best validation Dice:",
        round(summary["best_validation_dice"], 4),
    )
    print(
        "Best validation IoU:",
        round(summary["best_validation_iou"], 4),
    )
    print(
        "Best validation precision:",
        round(summary["best_validation_precision"], 4),
    )
    print(
        "Best validation recall:",
        round(summary["best_validation_recall"], 4),
    )

    print("\nBest model:", best_model_path)
    print("Training history:", history_path)
    print("Experiment summary:", summary_path)

    print(
        "\nThe held-out test set was not loaded or used."
    )


# ============================================================
# 11. COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare COVID infection U-Net loss functions."
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "baseline",
            "weighted_bce_dice",
            "tversky",
        ],
        help="Select one loss-function experiment.",
    )

    args = parser.parse_args()

    main(args.experiment)



# do baar run hooga ye sequence jaisa h vaisa h krna

# python .\scripts\112_compare_infection_loss_functions.py --experiment weighted_bce_dice
# python .\scripts\112_compare_infection_loss_functions.py --experiment tversky