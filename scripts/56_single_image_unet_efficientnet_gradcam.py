"""
Single-image Pneumonia CDSS inference prototype.

Runs:
1. Balanced U-Net lung segmentation.
2. Frozen original EfficientNetB0 classification.
3. Validation-selected Viral decision rule (P(Viral) >= 0.25).
4. Grad-CAM for the final selected class.

IMPORTANT:
- This script does NOT retrain or modify either checkpoint.
- U-Net predicts lung regions, not pneumonia/infection regions.
- Grad-CAM shows model-discriminative attention, not a validated lesion mask.
- Therefore this script deliberately does NOT report an infection percentage.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    PROJECT_ROOT / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

"""UNET_PATH = (
    PROJECT_ROOT / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)
"""  ## ye hele rhega uske baad 72 k time pe

UNET_PATH = (
    PROJECT_ROOT / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "classification" / "single_image_inference"

IMAGE_SIZE = 224
LUNG_THRESHOLD = 0.30
VIRAL_THRESHOLD = 0.25
GRADCAM_LAYER = "block5c_project_conv"

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]
VIRAL_INDEX = 1


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )
    # Keep 0..255. EfficientNetB0 includes its own rescaling.
    return tf.cast(image, tf.float32).numpy()


def load_grayscale_image(image_path: Path) -> np.ndarray:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )
    image.set_shape([None, None, 1])
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )
    # U-Net expects grayscale normalized to 0..1.
    return (tf.cast(image, tf.float32) / 255.0).numpy()


def choose_class(probabilities: np.ndarray):
    """Apply the validation-selected Viral override rule."""
    viral_probability = float(probabilities[VIRAL_INDEX])

    if viral_probability >= VIRAL_THRESHOLD:
        return VIRAL_INDEX, "viral_threshold"

    non_viral_indices = [i for i in range(len(CLASS_NAMES)) if i != VIRAL_INDEX]
    selected_non_viral = max(
        non_viral_indices,
        key=lambda index: float(probabilities[index]),
    )
    return selected_non_viral, "non_viral_argmax"


def build_gradcam_model(classifier):
    backbone = classifier.get_layer("efficientnetb0")
    last_conv_layer = backbone.get_layer(GRADCAM_LAYER)

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output],
        name="single_image_gradcam_backbone",
    )

    gradcam_input = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="gradcam_input",
    )
    conv_outputs, backbone_output = backbone_grad_model(
        gradcam_input,
        training=False,
    )

    x = classifier.get_layer("global_average_pooling")(backbone_output)
    x = classifier.get_layer("classification_dropout")(
        x,
        training=False,
    )
    predictions = classifier.get_layer("classification_output")(x)

    grad_model = tf.keras.Model(
        inputs=gradcam_input,
        outputs=[conv_outputs, predictions],
        name="single_image_gradcam_model",
    )
    return grad_model


def generate_gradcam(grad_model, rgb_batch, target_class_index):
    image_tensor = tf.convert_to_tensor(rgb_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False,
        )
        target_scores = predictions[:, target_class_index]
        target_score_sum = tf.reduce_sum(target_scores)

    gradients = tape.gradient(target_score_sum, conv_outputs)
    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. Check the selected layer "
            "and classifier architecture."
        )

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cams = tf.reduce_sum(
        conv_outputs * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    cams = tf.nn.relu(cams)

    maximum = tf.reduce_max(cams, axis=(1, 2), keepdims=True)
    cams = tf.math.divide_no_nan(cams, maximum)
    cams = tf.image.resize(
        cams[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )
    return cams[0, ..., 0].numpy(), predictions[0].numpy()


def calculate_attention_metrics(cam, lung_mask):
    lung_pixels = lung_mask > 0
    background_pixels = ~lung_pixels

    lung_area = int(np.sum(lung_pixels))
    total_pixels = int(lung_mask.size)
    background_area = total_pixels - lung_area

    result = {
        "lung_mask_area_percent_of_image": (
            float(lung_area / total_pixels * 100)
        ),
        "gradcam_activation_inside_lung_percent": None,
        "gradcam_activation_outside_lung_percent": None,
        "mean_gradcam_inside_lung": None,
        "mean_gradcam_outside_lung": None,
        "attention_analysis_valid": False,
    }

    if lung_area == 0 or background_area == 0:
        return result

    total_activation = float(np.sum(cam))
    if total_activation <= 0:
        return result

    inside_activation = float(np.sum(cam[lung_pixels]))
    outside_activation = float(np.sum(cam[background_pixels]))

    result.update({
        "gradcam_activation_inside_lung_percent": (
            inside_activation / total_activation * 100
        ),
        "gradcam_activation_outside_lung_percent": (
            outside_activation / total_activation * 100
        ),
        "mean_gradcam_inside_lung": float(np.mean(cam[lung_pixels])),
        "mean_gradcam_outside_lung": float(np.mean(cam[background_pixels])),
        "attention_analysis_valid": True,
    })
    return result


def save_visualizations(
    image_rgb,
    lung_mask,
    cam,
    predicted_class,
    confidence_percent,
    output_dir,
):
    image_display = np.clip(image_rgb / 255.0, 0.0, 1.0)
    lung_mask_bool = lung_mask.astype(bool)

    # Lung mask overlay.
    lung_overlay = image_display.copy()
    tint = np.zeros_like(lung_overlay)
    tint[..., 1] = 1.0
    lung_overlay[lung_mask_bool] = (
        0.55 * lung_overlay[lung_mask_bool]
        + 0.45 * tint[lung_mask_bool]
    )

    # Grad-CAM overlay.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_display)
    ax.imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    ax.set_title(
        f"Grad-CAM: {predicted_class} ({confidence_percent:.2f}%)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_display)
    axes[0].set_title("Input chest X-ray")
    axes[1].imshow(lung_overlay)
    axes[1].set_title("U-Net lung mask overlay")
    axes[2].imshow(image_display)
    axes[2].imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    axes[2].set_title(f"Grad-CAM: {predicted_class}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_inference_visualization.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(
        output_dir / "lung_mask.png",
        lung_mask_bool.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run U-Net + EfficientNetB0 inference on one chest X-ray."
    )
    parser.add_argument(
        "image",
        help="Path to a chest X-ray image (PNG/JPG/JPEG).",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {CLASSIFIER_PATH}")
    if not UNET_PATH.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {UNET_PATH}")

    output_dir = OUTPUT_ROOT / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Input image:", image_path)
    print("Loading original EfficientNetB0 classifier...")
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)

    print("Loading COVID-domain fine-tuned U-Net...")
    unet = tf.keras.models.load_model(UNET_PATH, compile=False)

    print("Preprocessing image...")
    rgb_image = load_rgb_image(image_path)
    gray_image = load_grayscale_image(image_path)

    rgb_batch = np.expand_dims(rgb_image, axis=0).astype(np.float32)
    gray_batch = np.expand_dims(gray_image, axis=0).astype(np.float32)

    print("Running lung segmentation...")
    mask_probabilities = unet(gray_batch, training=False).numpy()
    if mask_probabilities.ndim != 4 or mask_probabilities.shape[-1] < 1:
        raise ValueError(
            f"Unexpected U-Net output shape: {mask_probabilities.shape}"
        )
    lung_mask = mask_probabilities[0, ..., 0] >= LUNG_THRESHOLD

    print("Running pneumonia classification...")
    probabilities = classifier(rgb_batch, training=False).numpy()[0]
    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError(
            f"Expected 4 class probabilities, got shape {probabilities.shape}"
        )

    predicted_index, decision_rule = choose_class(probabilities)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence_percent = float(probabilities[predicted_index] * 100)

    print("Building Grad-CAM model...")
    grad_model = build_gradcam_model(classifier)

    print(f"Generating Grad-CAM for selected class: {predicted_class}")
    cam, gradcam_probabilities = generate_gradcam(
        grad_model,
        rgb_batch,
        predicted_index,
    )

    # Ensure the Grad-CAM graph and direct classifier output agree.
    if not np.allclose(probabilities, gradcam_probabilities, atol=1e-4):
        raise RuntimeError(
            "Classifier and Grad-CAM graph outputs differ. "
            "Do not trust this Grad-CAM until the architecture is checked."
        )

    attention_metrics = calculate_attention_metrics(cam, lung_mask)

    save_visualizations(
        rgb_image,
        lung_mask,
        cam,
        predicted_class,
        confidence_percent,
        output_dir,
    )

    result = {
        "input_image": str(image_path),
        "classifier_checkpoint": str(CLASSIFIER_PATH),
        "unet_checkpoint": str(UNET_PATH),
        "class_probabilities_percent": {
            name: float(probabilities[index] * 100)
            for index, name in enumerate(CLASS_NAMES)
        },
        "predicted_class": predicted_class,
        "selected_class_probability_percent": confidence_percent,
        "decision_rule": decision_rule,
        "viral_probability_threshold": VIRAL_THRESHOLD,
        "lung_segmentation_threshold": LUNG_THRESHOLD,
        **attention_metrics,
        "interpretation_limitations": [
            "The U-Net mask represents lung regions, not pneumonia lesions.",
            "Grad-CAM represents model attention, not a validated infection mask.",
            "No infection percentage or severity stage is calculated by this script.",
            "This research prototype is not a standalone clinical diagnosis.",
        ],
        "output_directory": str(output_dir),
    }

    with open(output_dir / "inference_result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\n" + "=" * 64)
    print("SINGLE-IMAGE INFERENCE RESULT")
    print("=" * 64)
    for class_name, value in result["class_probabilities_percent"].items():
        print(f"{class_name:12s}: {value:7.3f}%")
    print(f"\nSelected class: {predicted_class}")
    print(f"Selected probability: {confidence_percent:.3f}%")
    print(f"Decision rule: {decision_rule}")
    print(
        "Lung mask area (% of full image): "
        f"{attention_metrics['lung_mask_area_percent_of_image']:.2f}%"
    )
    print(
        "Grad-CAM activation inside lung (%): "
        f"{attention_metrics['gradcam_activation_inside_lung_percent']}"
    )
    print("\nSaved outputs to:", output_dir)
    print("\nNote: lung mask area is NOT infection percentage.")


if __name__ == "__main__":
    main()
