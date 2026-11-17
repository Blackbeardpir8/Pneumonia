
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
from aiml.config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    SEGMENTATION_DATA_DIR,
    SEGMENTATION_RESULTS_DIR,
    SAVED_MODELS_DIR,
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BEST_MODEL_PATH = (
    SAVED_MODELS_DIR / "unet_lung_segmentation_balanced.keras"
)

LAST_MODEL_PATH = (
    SAVED_MODELS_DIR / "unet_lung_segmentation_balanced_last.keras"
)

HISTORY_PATH = (
    SEGMENTATION_RESULTS_DIR /
    "balanced_training_history.csv"
)


# ============================================================
# 1. LOAD DATA BY SOURCE
# ============================================================

def load_source_data(split):
    image_dir = SEGMENTATION_DATA_DIR / split / "images"
    mask_dir = SEGMENTATION_DATA_DIR / split / "masks"

    data = {
        "CHNCXR": {"images": [], "masks": []},
        "MCUCXR": {"images": [], "masks": []},
    }

    for image_path in sorted(image_dir.glob("*.png")):

        source = image_path.name.split("_")[0]

        if source not in data:
            continue

        mask_path = (
            mask_dir /
            f"{image_path.stem}_mask.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = tf.io.decode_image(
            tf.io.read_file(str(image_path)),
            channels=1,
            expand_animations=False,
        )

        mask = tf.io.decode_image(
            tf.io.read_file(str(mask_path)),
            channels=1,
            expand_animations=False,
        )

        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.cast(mask >= 0.5, tf.float32)

        tf.debugging.assert_equal(
            tf.shape(image)[:2],
            [IMAGE_HEIGHT, IMAGE_WIDTH],
        )

        tf.debugging.assert_equal(
            tf.shape(mask)[:2],
            [IMAGE_HEIGHT, IMAGE_WIDTH],
        )

        data[source]["images"].append(image.numpy())
        data[source]["masks"].append(mask.numpy())

    result = {}

    for source in data:
        if not data[source]["images"]:
            raise ValueError(f"No {source} images found in {split}.")

        result[source] = (
            np.asarray(data[source]["images"], dtype=np.float32),
            np.asarray(data[source]["masks"], dtype=np.float32),
        )

        print(
            f"{split} {source}: "
            f"{len(result[source][0])} images"
        )

    return result


# ============================================================
# 2. BALANCE TRAINING DATA
# ============================================================

def balance_training_data(source_data):

    X_chn, y_chn = source_data["CHNCXR"]
    X_mcu, y_mcu = source_data["MCUCXR"]

    target_count = max(len(X_chn), len(X_mcu))

    rng = np.random.default_rng(SEED)

    chn_indices = rng.choice(
        len(X_chn),
        size=target_count,
        replace=len(X_chn) < target_count,
    )

    mcu_indices = rng.choice(
        len(X_mcu),
        size=target_count,
        replace=len(X_mcu) < target_count,
    )

    X = np.concatenate(
        [X_chn[chn_indices], X_mcu[mcu_indices]],
        axis=0,
    )

    y = np.concatenate(
        [y_chn[chn_indices], y_mcu[mcu_indices]],
        axis=0,
    )

    permutation = rng.permutation(len(X))

    X = X[permutation]
    y = y[permutation]

    print("\nBALANCED TRAINING DATA")
    print("-" * 45)
    print("CHNCXR samples:", target_count)
    print("MCUCXR samples:", target_count)
    print("Total samples:", len(X))

    return X, y


# ============================================================
# 3. LIGHTWEIGHT U-NET
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


def build_lightweight_unet():

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
    )(dec1)

    return keras.Model(inputs, outputs)


# ============================================================
# 4. METRICS
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
# 5. TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("BALANCED U-NET SEGMENTATION EXPERIMENT")
    print("=" * 60)

    train_data = load_source_data("train")
    val_data = load_source_data("val")

    X_train, y_train = balance_training_data(train_data)

    # Validation data is NOT oversampled.
    X_val = np.concatenate(
        [
            val_data["CHNCXR"][0],
            val_data["MCUCXR"][0],
        ],
        axis=0,
    )

    y_val = np.concatenate(
        [
            val_data["CHNCXR"][1],
            val_data["MCUCXR"][1],
        ],
        axis=0,
    )

    print("\nValidation samples:", len(X_val))

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_lightweight_unet()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            dice_coefficient,
            iou_score,
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
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
            str(HISTORY_PATH)
        ),

        keras.callbacks.TerminateOnNaN(),
    ]

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

    model.save(LAST_MODEL_PATH)

    print("\nBALANCED TRAINING COMPLETED")
    print("-" * 45)
    print("Best model:", BEST_MODEL_PATH)
    print("Last model:", LAST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)

    best_epoch = (
        int(np.argmax(history.history["val_dice_coefficient"])) + 1
    )

    best_dice = max(history.history["val_dice_coefficient"])

    print("Best epoch:", best_epoch)
    print("Best validation Dice:", round(best_dice, 4))


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/19_train_balanced_segmentation.py