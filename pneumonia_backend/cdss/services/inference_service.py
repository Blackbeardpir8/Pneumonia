
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# services/inference_service.py
# parents[3] = main Code directory

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

LUNG_UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

INFECTION_UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "infection_loss_experiments"
    / "weighted_bce_dice"
    / "best_model.keras"
)

OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "media"
    / "cdss"
)

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]

IMAGE_SIZE = 224
INFECTION_IMAGE_SIZE = 256

LUNG_THRESHOLD = 0.30
INFECTION_THRESHOLD = 0.50
VIRAL_THRESHOLD = 0.25

VIRAL_INDEX = 1
GRADCAM_LAYER = "block5c_project_conv"


# ============================================================
# 2. LOAD MODELS ONLY ONCE
# ============================================================

@lru_cache(maxsize=1)
def load_models():

    for path in [
        CLASSIFIER_PATH,
        LUNG_UNET_PATH,
        INFECTION_UNET_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}"
            )

    print("Loading EfficientNetB0 classifier...")

    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH,
        compile=False,
    )

    print("Loading lung segmentation U-Net...")

    lung_unet = tf.keras.models.load_model(
        LUNG_UNET_PATH,
        compile=False,
    )

    print("Loading infection segmentation U-Net...")

    infection_unet = tf.keras.models.load_model(
        INFECTION_UNET_PATH,
        compile=False,
    )

    print("Building Grad-CAM model...")

    gradcam_model = build_gradcam_model(classifier)

    print("All CDSS models loaded successfully.")

    return (
        classifier,
        lung_unet,
        infection_unet,
        gradcam_model,
    )


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def load_image(image_path, size, channels):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=channels,
        expand_animations=False,
    )

    image.set_shape([None, None, channels])

    image = tf.image.resize(
        image,
        [size, size],
        method="bilinear",
    )

    image = tf.cast(image, tf.float32)

    if channels == 1:
        image = image / 255.0

    # RGB classifier input intentionally remains 0–255.
    return image.numpy()


# ============================================================
# 4. CLASSIFICATION DECISION
# ============================================================

def choose_class(probabilities):

    viral_probability = float(
        probabilities[VIRAL_INDEX]
    )

    if viral_probability >= VIRAL_THRESHOLD:
        return VIRAL_INDEX, "viral_threshold"

    non_viral_indices = [
        index
        for index in range(len(CLASS_NAMES))
        if index != VIRAL_INDEX
    ]

    selected_index = max(
        non_viral_indices,
        key=lambda index: float(probabilities[index]),
    )

    return selected_index, "non_viral_argmax"


# ============================================================
# 5. GRAD-CAM
# ============================================================

def build_gradcam_model(classifier):

    backbone = classifier.get_layer("efficientnetb0")

    last_conv_layer = backbone.get_layer(
        GRADCAM_LAYER
    )

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output,
        ],
    )

    gradcam_input = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    conv_outputs, backbone_output = backbone_grad_model(
        gradcam_input,
        training=False,
    )

    x = classifier.get_layer(
        "global_average_pooling"
    )(backbone_output)

    x = classifier.get_layer(
        "classification_dropout"
    )(x, training=False)

    predictions = classifier.get_layer(
        "classification_output"
    )(x)

    return tf.keras.Model(
        inputs=gradcam_input,
        outputs=[conv_outputs, predictions],
    )


def generate_gradcam(
    grad_model,
    rgb_batch,
    target_class_index,
):

    image_tensor = tf.convert_to_tensor(
        rgb_batch,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False,
        )

        target_scores = predictions[
            :, target_class_index
        ]

        target_score_sum = tf.reduce_sum(
            target_scores
        )

    gradients = tape.gradient(
        target_score_sum,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    cams = tf.reduce_sum(
        conv_outputs
        * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    cams = tf.nn.relu(cams)

    maximum = tf.reduce_max(
        cams,
        axis=(1, 2),
        keepdims=True,
    )

    cams = tf.math.divide_no_nan(
        cams,
        maximum,
    )

    cams = tf.image.resize(
        cams[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )

    return cams[0, ..., 0].numpy(), predictions[0].numpy()


# ============================================================
# 6. VISUALIZATION HELPERS
# ============================================================

def make_overlay(image_rgb, mask, color):

    image = np.clip(
        image_rgb / 255.0,
        0.0,
        1.0,
    )

    overlay = image.copy()

    mask = mask.astype(bool)

    tint = np.zeros_like(overlay)
    tint[:] = color

    overlay[mask] = (
        0.55 * overlay[mask]
        + 0.45 * tint[mask]
    )

    return np.clip(overlay, 0.0, 1.0)


def resize_mask_for_display(mask):

    resized = tf.image.resize(
        mask.astype(np.float32)[..., np.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE],
        method="nearest",
    )

    return resized.numpy()[..., 0] > 0.5


def save_visualizations(
    image_rgb,
    lung_mask,
    infection_mask,
    cam,
    predicted_class,
    confidence,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_display = np.clip(
        image_rgb / 255.0,
        0.0,
        1.0,
    )

    infection_display_mask = resize_mask_for_display(
        infection_mask
    )

    lung_overlay = make_overlay(
        image_rgb,
        lung_mask,
        (0.0, 1.0, 0.0),
    )

    infection_overlay = make_overlay(
        image_rgb,
        infection_display_mask,
        (1.0, 0.0, 0.0),
    )

    plt.imsave(
        output_dir / "input_image.png",
        image_display,
    )

    plt.imsave(
        output_dir / "lung_mask.png",
        lung_mask.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.imsave(
        output_dir / "infection_mask.png",
        infection_mask.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.imsave(
        output_dir / "lung_overlay.png",
        lung_overlay,
    )

    plt.imsave(
        output_dir / "infection_overlay.png",
        infection_overlay,
    )

    # Grad-CAM overlay
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(image_display)
    ax.imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Grad-CAM: {predicted_class} "
        f"({confidence:.2f}%)"
    )

    ax.axis("off")
    fig.tight_layout()

    fig.savefig(
        output_dir / "gradcam_overlay.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    plt.imsave(
        output_dir / "gradcam_heatmap.png",
        cam,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    # Combined visualization
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    axes[0, 0].imshow(image_display)
    axes[0, 0].set_title("Input chest X-ray")

    axes[0, 1].imshow(lung_overlay)
    axes[0, 1].set_title("Lung segmentation")

    axes[0, 2].imshow(infection_overlay)
    axes[0, 2].set_title(
        "Infection segmentation (experimental)"
    )

    axes[1, 0].imshow(image_display)
    axes[1, 0].imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        f"Grad-CAM: {predicted_class}"
    )

    axes[1, 1].imshow(lung_mask, cmap="gray")
    axes[1, 1].set_title("Lung binary mask")

    axes[1, 2].imshow(infection_mask, cmap="gray")
    axes[1, 2].set_title(
        "Infection binary mask (256x256)"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_dir / "combined_visualization.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 7. MAIN INFERENCE SERVICE
# ============================================================

def run_inference(image_path, output_dir=None):

    image_path = Path(image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if output_dir is None:
        output_dir = (
            OUTPUT_ROOT / image_path.stem
        )

    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        classifier,
        lung_unet,
        infection_unet,
        gradcam_model,
    ) = load_models()

    # Separate preprocessing for each model.
    rgb_image = load_image(
        image_path,
        IMAGE_SIZE,
        channels=3,
    )

    gray_image = load_image(
        image_path,
        IMAGE_SIZE,
        channels=1,
    )

    infection_image = load_image(
        image_path,
        INFECTION_IMAGE_SIZE,
        channels=1,
    )

    rgb_batch = np.expand_dims(
        rgb_image,
        axis=0,
    ).astype(np.float32)

    gray_batch = np.expand_dims(
        gray_image,
        axis=0,
    ).astype(np.float32)

    infection_batch = np.expand_dims(
        infection_image,
        axis=0,
    ).astype(np.float32)

    # Lung segmentation
    lung_probabilities = lung_unet(
        gray_batch,
        training=False,
    ).numpy()

    lung_mask = (
        lung_probabilities[0, ..., 0]
        >= LUNG_THRESHOLD
    )

    # Infection segmentation
    infection_probabilities = infection_unet(
        infection_batch,
        training=False,
    ).numpy()

    infection_mask = (
        infection_probabilities[0, ..., 0]
        >= INFECTION_THRESHOLD
    )

    # Classification
    probabilities = classifier(
        rgb_batch,
        training=False,
    ).numpy()[0]

    predicted_index, decision_rule = choose_class(
        probabilities
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    # Raw highest-probability class (ordinary argmax)
    highest_probability_index = int(
        np.argmax(probabilities)
    )

    highest_probability_class = CLASS_NAMES[
        highest_probability_index
    ]

    highest_probability_percent = float(
        probabilities[highest_probability_index] * 100
    )

    # Check whether the decision rule overrode ordinary argmax
    decision_rule_overrode_argmax = (
        predicted_index != highest_probability_index
    )

    # Grad-CAM
    cam, gradcam_probabilities = generate_gradcam(
        gradcam_model,
        rgb_batch,
        predicted_index,
    )

    if not np.allclose(
        probabilities,
        gradcam_probabilities,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Grad-CAM and classifier predictions do not match."
        )

    # Measurements
    lung_area_pixels = int(lung_mask.sum())
    infection_area_pixels = int(infection_mask.sum())

    lung_area_percent = float(
        lung_mask.mean() * 100
    )

    infection_area_percent = float(
        infection_mask.mean() * 100
    )

    lung_activation = None
    background_activation = None

    if (
        lung_mask.any()
        and (~lung_mask).any()
        and cam.sum() > 0
    ):

        lung_activation = float(
            cam[lung_mask].sum()
            / cam.sum() * 100
        )

        background_activation = float(
            cam[~lung_mask].sum()
            / cam.sum() * 100
        )

    # Save visualizations
    save_visualizations(
        rgb_image,
        lung_mask,
        infection_mask,
        cam,
        predicted_class,
        confidence,
        output_dir,
    )

    # Structured result
    result = {
        "image_name": image_path.name,

        # Final model decision
        "predicted_class": predicted_class,
        "selected_class_probability_percent": confidence,

        # Raw highest-probability class
        "highest_probability_class": highest_probability_class,
        "highest_probability_percent": highest_probability_percent,

        # Decision rule transparency
        "decision_rule": decision_rule,
        "decision_rule_overrode_argmax": decision_rule_overrode_argmax,

        "class_probabilities_percent": {
            name: float(probabilities[index] * 100)
            for index, name in enumerate(CLASS_NAMES)
        },

        "lung_segmentation": {
            "threshold": LUNG_THRESHOLD,
            "mask_area_pixels": lung_area_pixels,
            "mask_area_percent_of_224_input": lung_area_percent,
            "mask_path": str(output_dir / "lung_mask.png"),
        },

        "infection_segmentation": {
            "threshold": INFECTION_THRESHOLD,
            "mask_area_pixels": infection_area_pixels,
            "mask_area_percent_of_256_input": infection_area_percent,
            "mask_path": str(output_dir / "infection_mask.png"),
            "status": "experimental",
            "validated_for_non_covid_classes": False,
        },

        "gradcam": {
            "target_class": predicted_class,
            "activation_inside_lung_percent": lung_activation,
            "activation_outside_lung_percent": background_activation,
            "overlay_path": str(output_dir / "gradcam_overlay.png"),
        },

        "visualizations": {
            "input": str(output_dir / "input_image.png"),
            "lung_overlay": str(output_dir / "lung_overlay.png"),
            "infection_overlay": str(output_dir / "infection_overlay.png"),
            "gradcam_heatmap": str(output_dir / "gradcam_heatmap.png"),
            "combined": str(output_dir / "combined_visualization.png"),
        },

        "limitations": [
            "Research prototype; not a standalone clinical diagnosis.",
            "Classifier probabilities are not guaranteed to be calibrated.",
            "Grad-CAM is model attribution, not a validated lesion map.",
            "Infection segmentation was trained using COVID-19 infection annotations.",
            "Infection mask area is not a clinically validated infection percentage.",
            "No clinical severity stage is calculated.",
        ],
    }

    with open(
        output_dir / "inference_result.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    return result
