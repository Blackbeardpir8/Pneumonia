
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import IMAGE_SIZE


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "dataset_source_classifier.keras"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_classifier"
    / "source_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_classifier"
    / "gradcam"
)

RANDOM_SEED = 42
MAX_IMAGES_PER_GROUP = 12
COLUMNS = 3
OVERLAY_ALPHA = 0.40

SOURCE_NAMES = [
    "bacterial_viral_dataset",
    "covid19_radiography",
]

SOURCE_TO_INDEX = {
    source: index
    for index, source in enumerate(SOURCE_NAMES)
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# LOAD MODEL AND PREDICTIONS
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Source classifier not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Source classifier loaded successfully.")

    return model


def load_predictions():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "processed_path",
        "actual_source",
        "predicted_source",
        "source_probability",
        "correct_prediction",
        "class_name",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    df["correct_prediction"] = (
        df["correct_prediction"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if df["correct_prediction"].isnull().any():
        raise ValueError(
            "Invalid values in correct_prediction column."
        )

    return df


# --------------------------------------------------
# FIND LAST CONVOLUTIONAL LAYER
# --------------------------------------------------

def find_last_conv_layer(model):

    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        )
    ]

    if not conv_layers:
        raise ValueError(
            "No Conv2D layer found in the source classifier."
        )

    last_conv_layer = conv_layers[-1]

    print(
        f"Grad-CAM convolutional layer: "
        f"{last_conv_layer.name}"
    )

    return last_conv_layer


# --------------------------------------------------
# BUILD GRAD-CAM MODEL
# --------------------------------------------------

def build_gradcam_model(model, last_conv_layer):

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output,
        ]
    )


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    model_input = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array, model_input


# --------------------------------------------------
# GENERATE GRAD-CAM
# --------------------------------------------------

def generate_gradcam(
    gradcam_model,
    image_input,
    target_source_index
):

    image_tensor = tf.convert_to_tensor(
        image_input,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = gradcam_model(
            image_tensor,
            training=False
        )

        source_probability = predictions[:, 0]

        if target_source_index == 1:
            target_score = source_probability
        else:
            target_score = 1.0 - source_probability

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

    conv_outputs = conv_outputs[0]
    gradients = gradients[0]

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + 1e-8),
        tf.zeros_like(heatmap)
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    return heatmap.numpy().squeeze()


# --------------------------------------------------
# CREATE CONTACT SHEET
# --------------------------------------------------

def create_contact_sheet(
    group_df,
    gradcam_model,
    actual_source,
    correctly_classified
):

    if group_df.empty:
        print(
            f"No examples: {actual_source}, "
            f"correct={correctly_classified}"
        )
        return

    group_df = group_df.sample(
        n=min(
            len(group_df),
            MAX_IMAGES_PER_GROUP
        ),
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    image_count = len(group_df)

    rows = (
        image_count + COLUMNS - 1
    ) // COLUMNS

    fig, axes = plt.subplots(
        rows,
        COLUMNS,
        figsize=(15, rows * 5)
    )

    axes = np.asarray(axes).reshape(-1)

    for index, ax in enumerate(axes):

        if index >= image_count:
            ax.axis("off")
            continue

        record = group_df.iloc[index]

        try:

            original_image, image_input = load_image(
                record["processed_path"]
            )

            predicted_source = record["predicted_source"]

            target_index = SOURCE_TO_INDEX[
                predicted_source
            ]

            heatmap = generate_gradcam(
                gradcam_model,
                image_input,
                target_index
            )

            probability = float(
                record["source_probability"]
            )

            # For class 0, the probability is 1 - sigmoid output.
            predicted_confidence = (
                probability
                if target_index == 1
                else 1.0 - probability
            )

            ax.imshow(
                original_image.astype(np.uint8)
            )

            ax.imshow(
                heatmap,
                cmap="jet",
                alpha=OVERLAY_ALPHA,
                vmin=0,
                vmax=1
            )

            status = (
                "CORRECT"
                if record["correct_prediction"]
                else "INCORRECT"
            )

            ax.set_title(
                f"Actual: {actual_source}\n"
                f"Predicted: {predicted_source}\n"
                f"Confidence: {predicted_confidence:.3f}\n"
                f"Status: {status}",
                fontsize=8
            )

            ax.axis("off")

        except Exception as error:

            ax.set_title(
                f"Image error:\n{str(error)[:80]}",
                fontsize=8
            )

            ax.axis("off")

    status = (
        "correct"
        if correctly_classified
        else "incorrect"
    )

    fig.suptitle(
        f"Source: {actual_source} | "
        f"Classification: {status}",
        fontsize=14
    )

    plt.tight_layout()

    filename = (
        f"{actual_source}_{status}_gradcam.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading source classifier...")

    model = load_model()

    predictions_df = load_predictions()

    last_conv_layer = find_last_conv_layer(
        model
    )

    gradcam_model = build_gradcam_model(
        model,
        last_conv_layer
    )

    print("\nTotal validation images:", len(predictions_df))

    selected_records = []

    for source in SOURCE_NAMES:

        source_df = predictions_df[
            predictions_df["actual_source"] == source
        ]

        for is_correct in [True, False]:

            group_df = source_df[
                source_df["correct_prediction"]
                == is_correct
            ]

            print(
                f"\nSource: {source}"
                f"\nCorrect: {is_correct}"
                f"\nAvailable images: {len(group_df)}"
            )

            create_contact_sheet(
                group_df,
                gradcam_model,
                source,
                is_correct
            )

            if not group_df.empty:

                selected = group_df.sample(
                    n=min(
                        len(group_df),
                        MAX_IMAGES_PER_GROUP
                    ),
                    random_state=RANDOM_SEED
                )

                selected_records.append(selected)

    if selected_records:

        selected_df = pd.concat(
            selected_records,
            ignore_index=True
        )

        selected_df.to_csv(
            OUTPUT_DIR / "gradcam_selected_examples.csv",
            index=False
        )

    print("\n" + "=" * 55)
    print("SOURCE CLASSIFIER GRAD-CAM COMPLETED")
    print("=" * 55)

    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
