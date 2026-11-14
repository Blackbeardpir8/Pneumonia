
import sys
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    SAVED_MODELS_DIR,
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_CHANNELS = 3

NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAME = "efficientnetb0_pneumonia_classifier"

MODEL_PATH = (
    SAVED_MODELS_DIR / f"{MODEL_NAME}.keras"
)

MODEL_RESULTS_DIR = (
    CLASSIFICATION_RESULTS_DIR / "model"
)

DROPOUT_RATE = 0.30
INITIAL_LEARNING_RATE = LEARNING_RATE

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

def create_data_augmentation():

    augmentation = models.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal",
                seed=RANDOM_SEED,
            ),

            layers.RandomRotation(
                factor=0.03,
                seed=RANDOM_SEED + 1,
            ),

            layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                seed=RANDOM_SEED + 2,
            ),
        ],
        name="data_augmentation",
    )

    return augmentation


# ============================================================
# 4. BUILD EFFICIENTNETB0 MODEL
# ============================================================

def build_efficientnet_model():

    print("\nLoading pretrained EfficientNetB0...")

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ),
    )

    # Freeze pretrained feature extractor initially.
    base_model.trainable = False

    print("EfficientNetB0 loaded successfully.")
    print("Pretrained backbone frozen.")

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ),
        name="chest_xray_input",
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    augmentation = create_data_augmentation()

    x = augmentation(inputs)

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    # Keep Batch Normalization layers in inference mode.
    x = base_model(
        x,
        training=False,
    )

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE,
        name="classification_dropout",
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification_output",
    )(x)

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME,
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(
        optimizer=Adam(
            learning_rate=INITIAL_LEARNING_RATE
        ),

        loss=SparseCategoricalCrossentropy(),

        metrics=[
            "accuracy",
        ],
    )

    return model, base_model


# ============================================================
# 5. SAVE MODEL AND CONFIGURATION
# ============================================================

def save_model_and_configuration(model):

    SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Model already exists: {MODEL_PATH}. "
            "Rename or archive it before creating another model."
        )

    # Save model architecture and initial weights.
    model.save(MODEL_PATH)

    # Save model summary.
    summary_path = (
        MODEL_RESULTS_DIR / "model_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )

    # Save model configuration.
    configuration = {
        "model_name": MODEL_NAME,
        "architecture": "EfficientNetB0",
        "pretrained_weights": "ImageNet",
        "input_shape": [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ],
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "output_activation": "softmax",
        "loss": "SparseCategoricalCrossentropy",
        "optimizer": "Adam",
        "learning_rate": INITIAL_LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "backbone_trainable": False,
        "augmentation": [
            "Random horizontal flip",
            "Random rotation",
            "Random zoom",
        ],
        "random_seed": RANDOM_SEED,
        "training_status": "Model built; not trained yet.",
    }

    config_path = (
        MODEL_RESULTS_DIR / "model_configuration.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
        )

    print("\nModel saved:", MODEL_PATH)
    print("Model summary:", summary_path)
    print("Configuration:", config_path)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PNEUMONIA CDSS - EFFICIENTNETB0 MODEL BUILDING")
    print("=" * 65)

    print("TensorFlow version:", tf.__version__)

    print("Input image size:", IMAGE_SIZE)
    print("Classes:", CLASS_NAMES)

    model, base_model = build_efficientnet_model()

    print("\nMODEL SUMMARY")
    print("-" * 65)

    model.summary()

    total_parameters = model.count_params()

    trainable_parameters = sum(
        tf.size(variable).numpy()
        for variable in model.trainable_variables
    )

    print("\nTotal parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)
    print("Frozen backbone:", not base_model.trainable)

    save_model_and_configuration(model)

    print("\n" + "=" * 65)
    print("EFFICIENTNETB0 MODEL CREATED SUCCESSFULLY")
    print("Training has NOT started yet.")
    print("=" * 65)


if __name__ == "__main__":
    main()
