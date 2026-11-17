
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
    RANDOM_SEED,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnetb0_pneumonia_classifier"

INITIAL_MODEL_PATH = (
    SAVED_MODELS_DIR
    / f"{MODEL_NAME}_best.keras"
)

FINETUNED_MODEL_PATH = (
    SAVED_MODELS_DIR
    / f"{MODEL_NAME}_finetuned_best.keras"
)

MANIFEST_PATH = (
    CLASSIFICATION_DATA_DIR / "manifest.csv"
)

CLASS_WEIGHTS_PATH = (
    CLASSIFICATION_RESULTS_DIR
    / "pipeline"
    / "class_weights.json"
)

RESULTS_DIR = (
    CLASSIFICATION_RESULTS_DIR
    / "fine_tuning"
)

HISTORY_DIR = RESULTS_DIR / "history"

IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE

MAX_EPOCHS = 10
FINE_TUNE_LEARNING_RATE = 1e-5

EARLY_STOPPING_PATIENCE = 4
LR_REDUCTION_PATIENCE = 2

AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# 3. VALIDATE INPUT FILES
# ============================================================

def validate_files():

    required_files = [
        INITIAL_MODEL_PATH,
        MANIFEST_PATH,
        CLASS_WEIGHTS_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    if FINETUNED_MODEL_PATH.exists():

        raise FileExistsError(
            f"Fine-tuned checkpoint already exists: "
            f"{FINETUNED_MODEL_PATH}. "
            "Use a new filename or archive the previous experiment."
        )


# ============================================================
# 4. LOAD TRAINING METADATA
# ============================================================

def load_training_metadata():

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "split",
        "class_name",
        "class_index",
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

    # Verify all expected splits.
    for split in ["train", "val", "test"]:

        if not (df["split"] == split).any():
            raise ValueError(
                f"Missing {split} split."
            )

    # Verify class mapping.
    for _, row in df.iterrows():

        expected_index = CLASS_TO_INDEX.get(
            row["class_name"]
        )

        if expected_index is None:
            raise ValueError(
                f"Unknown class: {row['class_name']}"
            )

        if int(row["class_index"]) != expected_index:
            raise ValueError(
                f"Class mapping mismatch: {row['class_name']}"
            )

    # Load existing class weights.
    with open(
        CLASS_WEIGHTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        saved_weights = json.load(file)

    class_weights = {
        CLASS_TO_INDEX[class_name]: float(
            saved_weights[class_name]
        )
        for class_name in CLASS_NAMES
    }

    print("\nDataset metadata loaded.")

    print("Training images:", len(df[df["split"] == "train"]))
    print("Validation images:", len(df[df["split"] == "val"]))

    print("\nClass weights:")

    for class_name in CLASS_NAMES:

        index = CLASS_TO_INDEX[class_name]

        print(
            f"{class_name}: {class_weights[index]:.4f}"
        )

    return df, class_weights


# ============================================================
# 5. IMAGE PREPROCESSING
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

    # Keep image values in the 0-255 range.
    # EfficientNetB0 includes its own rescaling layer.

    return image, label


# ============================================================
# 6. BUILD DATASETS
# ============================================================

def build_dataset(df, split, training=False):

    split_df = df[
        df["split"] == split
    ].copy()

    if split_df.empty:
        raise ValueError(
            f"No images available for {split}."
        )

    image_paths = []

    for path in split_df["processed_path"].astype(str):

        image_path = Path(path)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image_paths.append(str(image_path))

    labels = split_df[
        "class_index"
    ].astype(np.int32).to_numpy()

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
        "images",
    )

    return dataset


# ============================================================
# 7. LOAD AND PREPARE FINE-TUNING MODEL
# ============================================================

def load_and_prepare_model():

    print("\nLoading existing best classifier...")

    model = tf.keras.models.load_model(
        INITIAL_MODEL_PATH,
        compile=False,
    )

    backbone = model.get_layer("efficientnetb0")

    # Unfreeze the backbone first.
    backbone.trainable = True

    trainable_prefixes = (
        "block6c_",
        "block6d_",
        "block7a_",
        "top_conv",
    )

    trainable_layer_names = []

    # Selectively unfreeze the later EfficientNet layers.
    for layer in backbone.layers:

        selected = layer.name.startswith(
            trainable_prefixes
        )

        is_batch_norm = isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )

        # Keep BatchNormalization frozen.
        if selected and not is_batch_norm:

            layer.trainable = True
            trainable_layer_names.append(layer.name)

        else:

            layer.trainable = False

    # Keep classification head trainable.
    for layer_name in [
        "global_average_pooling",
        "classification_dropout",
        "classification_output",
    ]:

        model.get_layer(layer_name).trainable = True

    # Recompile after changing trainability.
    model.compile(
        optimizer=Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("\nFine-tuning configuration:")

    print("Backbone:", backbone.name)

    print(
        "Selected trainable backbone layers:",
        len(trainable_layer_names),
    )

    print("\nSelected layers:")

    for name in trainable_layer_names:
        print("-", name)

    print(
        "\nTrainable parameters:",
        sum(
            int(np.prod(variable.shape))
            for variable in model.trainable_variables
        ),
    )

    print(
        "Total parameters:",
        model.count_params(),
    )

    return model


# ============================================================
# 8. CREATE CALLBACKS
# ============================================================

def create_callbacks():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [

        ModelCheckpoint(
            filepath=FINETUNED_MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=LR_REDUCTION_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks


# ============================================================
# 9. SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):

    history_df = pd.DataFrame(
        history.history
    )

    csv_path = (
        HISTORY_DIR / "finetuning_history.csv"
    )

    json_path = (
        HISTORY_DIR / "finetuning_history.json"
    )

    history_df.to_csv(
        csv_path,
        index=False,
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
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PNEUMONIA CDSS - EFFICIENTNETB0 FINE-TUNING")
    print("=" * 70)

    print("TensorFlow:", tf.__version__)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Learning rate:", FINE_TUNE_LEARNING_RATE)

    # Step 1: Validate files.
    validate_files()

    # Step 2: Load metadata.
    df, class_weights = load_training_metadata()

    # Step 3: Build training and validation datasets.
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

    # Test data deliberately not loaded.

    # Step 4: Load existing best model.
    model = load_and_prepare_model()

    # Step 5: Evaluate starting model on validation data.
    print("\nEvaluating starting checkpoint...")

    initial_results = model.evaluate(
        val_ds,
        return_dict=True,
        verbose=1,
    )

    print("\nStarting validation metrics:")
    print(initial_results)

    # Step 6: Prepare callbacks.
    callbacks = create_callbacks()

    # Step 7: Fine-tune.
    print("\nStarting fine-tuning...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 8: Save history.
    save_training_history(history)

    # Step 9: Evaluate restored best weights.
    print("\nEvaluating fine-tuned model...")

    final_results = model.evaluate(
        val_ds,
        return_dict=True,
        verbose=1,
    )

    print("\nFinal validation metrics:")
    print(final_results)

    # Step 10: Save experiment metadata.
    metadata = {
        "starting_checkpoint": str(INITIAL_MODEL_PATH),
        "fine_tuned_checkpoint": str(FINETUNED_MODEL_PATH),
        "architecture": "EfficientNetB0",
        "unfrozen_blocks": [
            "block6c",
            "block6d",
            "block7a",
            "top_conv",
        ],
        "batch_normalization_frozen": True,
        "learning_rate": FINE_TUNE_LEARNING_RATE,
        "maximum_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "class_names": CLASS_NAMES,
        "initial_validation": initial_results,
        "final_validation": final_results,
        "test_set_used": False,
    }

    metadata_path = (
        RESULTS_DIR / "finetuning_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nFine-tuned checkpoint:")
    print(FINETUNED_MODEL_PATH)

    print("\nMetadata:")
    print(metadata_path)

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETED")
    print("Original best checkpoint was not overwritten.")
    print("Test dataset was not used.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/40_finetune_efficientnet.py