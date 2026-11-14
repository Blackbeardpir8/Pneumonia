import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_evaluation"
    / "validation_predictions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification/gradcam_preview"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Normal",
    "Viral",
    "Bacterial",
    "COVID",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

LAST_CONV_LAYER = "block5c_project_conv"


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def load_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    # Keep values in the 0-255 range.
    # EfficientNet performs internal rescaling.
    image = tf.cast(image, tf.float32)

    return image.numpy()


# =========================================================
# SELECT ONE SAMPLE PER CLASS
# =========================================================

def select_samples():

    df = pd.read_csv(VALIDATION_CSV)

    selected = []

    for class_name in CLASS_NAMES:

        class_df = df[
            df["actual_class"] == class_name
        ]

        if class_df.empty:
            raise ValueError(
                f"No validation sample found for {class_name}"
            )

        # Select the first correctly classified sample.
        correct_df = class_df[
            class_df["predicted_class"] == class_name
        ]

        if correct_df.empty:
            raise ValueError(
                f"No correctly classified sample found for {class_name}"
            )

        selected.append(correct_df.iloc[0])

    return selected


# =========================================================
# BUILD GRAD-CAM MODEL
# =========================================================


def build_gradcam_model(model):
    """
    Build Grad-CAM model using the nested EfficientNetB0 backbone.
    """

    print("\nBuilding Grad-CAM model...")

    # Get nested EfficientNetB0
    backbone = model.get_layer("efficientnetb0")

    # Get final convolutional layer
    last_conv_layer = backbone.get_layer(LAST_CONV_LAYER)

    # Create intermediate model
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ],
        name="efficientnet_gradcam_backbone"
    )

    # New input for Grad-CAM
    gradcam_input = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="gradcam_input"
    )

    # Get convolutional features and backbone output
    conv_outputs, backbone_output = backbone_grad_model(
        gradcam_input,
        training=False
    )

    # Reuse the original classification head
    x = model.get_layer("global_average_pooling")(
        backbone_output
    )

    x = model.get_layer("classification_dropout")(
        x,
        training=False
    )

    predictions = model.get_layer("classification_output")(x)

    # Final Grad-CAM model
    grad_model = tf.keras.Model(
        inputs=gradcam_input,
        outputs=[conv_outputs, predictions],
        name="gradcam_model"
    )

    print("Grad-CAM model built successfully.")
    print("Last convolutional layer:", last_conv_layer.name)
    print("Feature map shape:", last_conv_layer.output.shape)

    return grad_model



# =========================================================
# GENERATE GRAD-CAM
# =========================================================

def generate_gradcam(
    grad_model,
    image_batch,
    class_index
):

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        target_score = predictions[:, class_index]

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        conv_outputs
        * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    cam = tf.nn.relu(cam)

    maximum = tf.reduce_max(
        cam,
        axis=(1, 2),
        keepdims=True
    )

    cam = tf.math.divide_no_nan(
        cam,
        maximum
    )

    cam = tf.image.resize(
        cam[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    return (
        cam[0, ..., 0].numpy(),
        predictions.numpy()[0]
    )


# =========================================================
# SAVE VISUALIZATION
# =========================================================

def save_visualization(
    image,
    heatmap,
    probabilities,
    actual_class,
    index
):

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    image_normalized = np.clip(
        image / 255.0,
        0,
        1
    )

    heatmap_colored = plt.get_cmap("jet")(
        heatmap
    )[..., :3]

    overlay = (
        0.55 * image_normalized
        + 0.45 * heatmap_colored
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10)
    )

    axes[0, 0].imshow(image_normalized)
    axes[0, 0].set_title("Original X-ray")

    axes[0, 1].imshow(heatmap, cmap="jet")
    axes[0, 1].set_title("Grad-CAM Heatmap")

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Grad-CAM Overlay")

    axes[1, 1].axis("off")

    probability_text = "\n".join(
        [
            f"{name}: {probabilities[i] * 100:.2f}%"
            for i, name in enumerate(CLASS_NAMES)
        ]
    )

    axes[1, 1].text(
        0.05,
        0.90,
        f"Actual class: {actual_class}\n"
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%\n\n"
        f"Class probabilities:\n{probability_text}",
        fontsize=12,
        verticalalignment="top"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Grad-CAM - {actual_class}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        f"gradcam_{index}_{actual_class}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "confidence_percent": confidence,
        "correct_prediction": (
            actual_class == predicted_class
        ),
        "output_image": str(output_path),
    }


# =========================================================
# INSPECT EFFICIENTNET LAYERS
# =========================================================

def inspect_efficientnet_layers(model):
    backbone = model.get_layer("efficientnetb0")
    print("\nNested EfficientNetB0 layers:")
    for layer in backbone.layers:
        if "conv" in layer.name.lower():
            print(layer.name, layer.output.shape)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("PNEUMONIA CLASSIFIER - GRAD-CAM PREVIEW")
    print("=" * 60)

    print("\nLoading validation samples...")

    samples = select_samples()

    print(f"Selected samples: {len(samples)}")

    print("\nLoading EfficientNetB0 checkpoint...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    inspect_efficientnet_layers(model)

    print("\nBuilding Grad-CAM model...")

    grad_model = build_gradcam_model(model)

    results = []

    for index, sample in enumerate(
        samples,
        start=1
    ):

        actual_class = sample["actual_class"]

        image_path = Path(
            sample["processed_path"]
        )

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        print(
            f"\n[{index}/4] Processing {actual_class}"
        )

        image = load_image(image_path)

        image_batch = image[np.newaxis, ...]

        class_index = CLASS_TO_INDEX[
            actual_class
        ]

        heatmap, probabilities = generate_gradcam(
            grad_model,
            image_batch,
            class_index
        )

        result = save_visualization(
            image,
            heatmap,
            probabilities,
            actual_class,
            index
        )

        results.append(result)

        print(
            f"Predicted: {result['predicted_class']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence_percent']:.2f}%"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "gradcam_preview_summary.csv",
        index=False
    )

    print("\nGrad-CAM preview completed.")

    print(f"Output folder: {OUTPUT_DIR}")

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()