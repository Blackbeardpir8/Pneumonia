
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"

PIPELINE_RESULTS_DIR = (
    CLASSIFICATION_RESULTS_DIR / "pipeline"
)

IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
NUM_CLASSES = len(CLASS_NAMES)

AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 3. LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "split",
        "class_name",
        "class_index",
        "processed_path",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    expected_splits = {"train", "val", "test"}

    if set(df["split"].unique()) != expected_splits:
        raise ValueError(
            "Manifest must contain train, val and test splits."
        )

    if not set(df["class_name"].unique()).issubset(
        set(CLASS_NAMES)
    ):
        raise ValueError(
            "Unexpected class name found in manifest."
        )

    for _, row in df.iterrows():

        expected_index = CLASS_TO_INDEX[row["class_name"]]

        if int(row["class_index"]) != expected_index:
            raise ValueError(
                f"Incorrect class index for {row['class_name']}"
            )

        if not Path(row["processed_path"]).is_file():
            raise FileNotFoundError(
                f"Image not found: {row['processed_path']}"
            )

    print("\nManifest loaded successfully.")
    print("Total images:", len(df))

    return df


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path, label):

    image_data = tf.io.read_file(image_path)

    # Convert grayscale and RGB images into 3-channel RGB.
    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    # Resize every image to the configured dimensions.
    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    # EfficientNetB0 in tf.keras includes its own input
    # rescaling layer. Keep pixel values in the [0, 255] range.
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)

    return image, label


# ============================================================
# 5. BUILD TF.DATA DATASET
# ============================================================

def build_dataset(df, split, training=False):

    split_df = df[df["split"] == split].copy()

    if split_df.empty:
        raise ValueError(
            f"No images found for {split} split."
        )

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
        f"{split.capitalize()} dataset created:",
        len(split_df),
        "images"
    )

    return dataset


# ============================================================
# 6. CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(df):

    train_df = df[df["split"] == "train"]

    classes = np.array(
        sorted(CLASS_TO_INDEX.values())
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["class_index"].to_numpy(),
    )

    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, weights)
    }

    print("\nClass weights:")

    for class_name, class_index in CLASS_TO_INDEX.items():

        print(
            f"{class_name}: "
            f"{class_weights[class_index]:.4f}"
        )

    return class_weights


# ============================================================
# 7. VERIFY DATA PIPELINE
# ============================================================

def verify_pipeline(train_ds, val_ds, test_ds):

    print("\nVerifying dataset batches...")

    for split, dataset in [
        ("Training", train_ds),
        ("Validation", val_ds),
        ("Testing", test_ds),
    ]:

        images, labels = next(iter(dataset))

        print(f"\n{split} batch:")
        print("Image shape:", images.shape)
        print("Label shape:", labels.shape)
        print("Image dtype:", images.dtype)
        print(
            "Pixel range:",
            float(tf.reduce_min(images)),
            "to",
            float(tf.reduce_max(images)),
        )

        print(
            "Label values:",
            np.unique(labels.numpy())
        )

        if images.shape[1:] != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ):
            raise ValueError(
                f"Incorrect image shape in {split} dataset."
            )

        if not np.all(np.isfinite(images.numpy())):
            raise ValueError(
                f"Non-finite image values in {split} dataset."
            )

    print("\nAll dataset batches verified successfully.")


# ============================================================
# 8. SAVE PIPELINE REPORTS
# ============================================================

def save_pipeline_reports(
    df,
    class_weights,
):

    PIPELINE_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights_path = (
        PIPELINE_RESULTS_DIR / "class_weights.json"
    )

    with open(
        weights_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                CLASS_NAMES[index]: weight
                for index, weight in class_weights.items()
            },
            file,
            indent=4,
        )

    summary = {
        "total_images": int(len(df)),
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "image_channels": 3,
        "batch_size": BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "random_seed": RANDOM_SEED,
        "input_pixel_range": [0, 255],
        "image_preprocessing": [
            "Decode image",
            "Convert to RGB",
            "Resize to configured dimensions",
            "Convert to float32",
        ],
        "augmentation": (
            "Not applied yet; will be included "
            "in the training model."
        ),
        "split_counts": {
            split: int((df["split"] == split).sum())
            for split in ["train", "val", "test"]
        },
    }

    summary_path = (
        PIPELINE_RESULTS_DIR / "pipeline_summary.json"
    )

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

    print("\nPipeline reports saved:")
    print(weights_path)
    print(summary_path)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PNEUMONIA CDSS - EFFICIENTNETB0 DATA PIPELINE")
    print("=" * 65)

    print("\nTensorFlow version:", tf.__version__)

    df = load_manifest()

    train_ds = build_dataset(
        df,
        "train",
        training=True,
    )

    val_ds = build_dataset(
        df,
        "val",
    )

    test_ds = build_dataset(
        df,
        "test",
    )

    class_weights = calculate_class_weights(df)

    verify_pipeline(
        train_ds,
        val_ds,
        test_ds,
    )

    save_pipeline_reports(
        df,
        class_weights,
    )

    print("\n" + "=" * 65)
    print("CLASSIFICATION DATA PIPELINE READY")
    print("=" * 65)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/04_build_classification_pipeline.py