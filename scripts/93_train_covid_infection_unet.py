
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

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
)

BEST_MODEL_PATH = (
    SAVED_MODELS_DIR
    / "unet_covid_infection_segmentation_best.keras"
)

LAST_MODEL_PATH = (
    SAVED_MODELS_DIR
    / "unet_covid_infection_segmentation_last.keras"
)

HISTORY_PATH = RESULTS_DIR / "training_history.csv"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
MASK_THRESHOLD = 0.5

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 2. DATASET PATH COLLECTION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg"
}


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
            f"Extra masks={len(extra_masks)}"
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
# 3. IMAGE AND MASK PREPROCESSING
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
# DATASET PIPELINE
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
# 4. LIGHTWEIGHT U-NET ARCHITECTURE
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
# 5. LOSS FUNCTION AND METRICS
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


@keras.utils.register_keras_serializable(
    package="PneumoniaCDSS"
)
def combined_bce_dice_loss(y_true, y_pred):

    bce = keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )

    bce = tf.reduce_mean(bce)

    return bce + dice_loss(y_true, y_pred)


@keras.utils.register_keras_serializable(
    package="PneumoniaCDSS"
)
def dice_coefficient(y_true, y_pred):

    y_pred = tf.cast(
        y_pred >= MASK_THRESHOLD,
        tf.float32,
    )

    y_true = tf.cast(y_true, tf.float32)

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


@keras.utils.register_keras_serializable(
    package="PneumoniaCDSS"
)
def iou_score(y_true, y_pred):

    y_pred = tf.cast(
        y_pred >= MASK_THRESHOLD,
        tf.float32,
    )

    y_true = tf.cast(y_true, tf.float32)

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
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("COVID INFECTION SEGMENTATION - LIGHTWEIGHT U-NET")
    print("=" * 65)

    SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_pairs = collect_image_mask_pairs("Train")
    val_pairs = collect_image_mask_pairs("Val")

    print("\nLoading training pipeline...")
    train_dataset = create_dataset(
        train_pairs,
        training=True,
    )

    print("Loading validation pipeline...")
    val_dataset = create_dataset(
        val_pairs,
        training=False,
    )

    print("\nBuilding a new U-Net from scratch...")

    model = build_infection_unet()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss=combined_bce_dice_loss,
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
            filepath=str(BEST_MODEL_PATH),
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
            str(HISTORY_PATH),
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

    model.save(LAST_MODEL_PATH)

    best_epoch = (
        int(
            np.argmax(
                history.history["val_dice_coefficient"]
            )
        ) + 1
    )

    best_dice = max(
        history.history["val_dice_coefficient"]
    )

    print("\n" + "=" * 65)
    print("INFECTION SEGMENTATION TRAINING COMPLETED")
    print("=" * 65)

    print("Best epoch:", best_epoch)
    print("Best validation Dice:", round(best_dice, 4))

    print("Best model:", BEST_MODEL_PATH)
    print("Last model:", LAST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)

    print(
        "\nNOTE: The test dataset was not used during training "
        "or model selection."
    )


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/93_train_covid_infection_unet.py