
"""
Train a Lightweight U-Net for lung segmentation.

Dataset:
    data/processed/segmentation/
        train/
            images/
            masks/
        val/
            images/
            masks/
        test/
            images/
            masks/

The test split is kept separate and is not used during training
or model selection.
"""

import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from aiml.config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    SAVED_MODELS_DIR,
    SEGMENTATION_DATA_DIR,
    SEGMENTATION_RESULTS_DIR,
    UNET_MODEL_PATH,
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. DATASET LOADING
# ============================================================

def load_dataset(split):
    """
    Load image-mask pairs from a processed dataset split.

    X-ray images are normalized to [0, 1].
    Masks are converted to binary values: 0 or 1.
    """

    image_dir = SEGMENTATION_DATA_DIR / split / "images"
    mask_dir = SEGMENTATION_DATA_DIR / split / "masks"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing image or mask directory for split '{split}'."
        )

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
    )
    mask_paths = sorted(
        path for path in mask_dir.iterdir()
        if path.is_file()
    )

    # Mask filenames append "_mask" to the corresponding image stem.
    mask_by_name = {
        path.stem.removesuffix("_mask") + path.suffix: path
        for path in mask_paths
    }

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"{split}: image count ({len(image_paths)}) does not "
            f"match mask count ({len(mask_paths)})."
        )

    images = []
    masks = []

    for image_path in image_paths:
        mask_path = mask_by_name.get(image_path.name)

        if mask_path is None:
            raise ValueError(
                f"No corresponding mask found for {image_path.name} "
                f"in split '{split}'."
            )

        # Read the X-ray as a single grayscale channel.
        image_bytes = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(
            image_bytes,
            channels=1,
            expand_animations=False,
        )
        image = tf.cast(image, tf.float32) / 255.0

        # Read the mask as grayscale.
        mask_bytes = tf.io.read_file(str(mask_path))
        mask = tf.io.decode_image(
            mask_bytes,
            channels=1,
            expand_animations=False,
        )
        mask = tf.cast(mask, tf.float32) / 255.0

        # The preprocessing script should already have resized
        # both files. Assert dimensions rather than silently resizing.
        tf.debugging.assert_equal(
            tf.shape(image)[:2],
            [IMAGE_HEIGHT, IMAGE_WIDTH],
            message=f"Unexpected image dimensions: {image_path}",
        )
        tf.debugging.assert_equal(
            tf.shape(mask)[:2],
            [IMAGE_HEIGHT, IMAGE_WIDTH],
            message=f"Unexpected mask dimensions: {mask_path}",
        )

        # Convert masks to binary labels.
        mask = tf.cast(mask >= 0.5, tf.float32)

        images.append(image.numpy())
        masks.append(mask.numpy())

    if not images:
        raise ValueError(f"No image-mask pairs found in '{split}'.")

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(masks, dtype=np.float32)

    print(
        f"{split}: images={X.shape}, masks={y.shape}, "
        f"foreground_fraction={np.mean(y):.4f}"
    )

    return X, y


# ============================================================
# 3. LIGHTWEIGHT U-NET ARCHITECTURE
# ============================================================

def convolution_block(inputs, filters):
    """Two convolution layers with batch normalization."""

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def build_lightweight_unet():
    """Build a compact U-Net with a one-channel input and output."""

    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        name="xray_input",
    )

    # Encoder
    enc1 = convolution_block(inputs, 16)
    pool1 = layers.MaxPooling2D(pool_size=2)(enc1)

    enc2 = convolution_block(pool1, 32)
    pool2 = layers.MaxPooling2D(pool_size=2)(enc2)

    enc3 = convolution_block(pool2, 64)
    pool3 = layers.MaxPooling2D(pool_size=2)(enc3)

    # Bottleneck
    bottleneck = convolution_block(pool3, 128)

    # Decoder
    up3 = layers.UpSampling2D(size=2, interpolation="bilinear")(
        bottleneck
    )
    up3 = layers.Concatenate()([up3, enc3])
    dec3 = convolution_block(up3, 64)

    up2 = layers.UpSampling2D(size=2, interpolation="bilinear")(
        dec3
    )
    up2 = layers.Concatenate()([up2, enc2])
    dec2 = convolution_block(up2, 32)

    up1 = layers.UpSampling2D(size=2, interpolation="bilinear")(
        dec2
    )
    up1 = layers.Concatenate()([up1, enc1])
    dec1 = convolution_block(up1, 16)

    outputs = layers.Conv2D(
        1,
        kernel_size=1,
        activation="sigmoid",
        name="lung_mask",
    )(dec1)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Lightweight_UNet",
    )


# ============================================================
# 4. SEGMENTATION METRICS
# ============================================================

@keras.utils.register_keras_serializable(package="PneumoniaCDSS")
def dice_coefficient(y_true, y_pred):
    """Calculate soft Dice coefficient."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2.0 * intersection + 1e-6) / (
        denominator + 1e-6
    )


@keras.utils.register_keras_serializable(package="PneumoniaCDSS")
def iou_score(y_true, y_pred):
    """Calculate soft Intersection over Union."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (intersection + 1e-6) / (union + 1e-6)


# ============================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("PNEUMONIA CDSS — LIGHTWEIGHT U-NET TRAINING")
    print("=" * 65)

    print("TensorFlow version:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())
    print("Dataset directory:", SEGMENTATION_DATA_DIR)

    # Load only training and validation data at this stage.
    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")

    # Ensure the expected dataset sizes are present.
    if len(X_train) != 492 or len(X_val) != 105:
        raise ValueError(
            "Unexpected dataset sizes. Expected 492 training and "
            f"105 validation pairs; found {len(X_train)} and "
            f"{len(X_val)}."
        )

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_lightweight_unet()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[dice_coefficient, iou_score],
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(UNET_MODEL_PATH),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(SEGMENTATION_RESULTS_DIR / "training_history.csv")
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Save the final in-memory model state separately from the
    # best checkpoint selected by validation Dice.
    final_model_path = (
        SAVED_MODELS_DIR / "unet_lung_segmentation_last.keras"
    )
    model.save(final_model_path)

    # Save training history as JSON for later analysis.
    history_path = (
        SEGMENTATION_RESULTS_DIR / "training_history.json"
    )
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )

    print("\nTraining completed.")
    print("Best checkpoint:", UNET_MODEL_PATH)
    print("Final model:", final_model_path)
    print("CSV history:", SEGMENTATION_RESULTS_DIR / "training_history.csv")
    print("JSON history:", history_path)
    print(
        "\nThe test split has not been loaded or used. "
        "We will evaluate it separately after checking training results."
    )


if __name__ == "__main__":
    main()


# to run this code, use the following command:
# python scripts/03_train_segmentation_model.py