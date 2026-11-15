
"""
Complete Pneumonia CDSS Inference Pipeline

Supports:
1. One chest X-ray image.
2. A folder of chest X-ray images.

Models:
- EfficientNetB0 pneumonia classifier
- COVID-domain fine-tuned lung U-Net
- Weighted BCE + Dice infection U-Net

Outputs:
- Classification probabilities
- Lung segmentation mask and overlay
- Infection segmentation mask and overlay
- Grad-CAM heatmap and overlay
- Combined visualization
- Individual JSON results
- Batch CSV summary

Research prototype only. Not for clinical diagnosis.
"""

import argparse
import json
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    PROJECT_ROOT / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

LUNG_UNET_PATH = (
    PROJECT_ROOT / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

INFECTION_UNET_PATH = (
    PROJECT_ROOT / "saved_models"
    / "infection_loss_experiments"
    / "weighted_bce_dice"
    / "best_model.keras"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "complete_cdss_inference"
)

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]

IMAGE_SIZE = 224
INFECTION_IMAGE_SIZE = 256

LUNG_THRESHOLD = 0.30
INFECTION_THRESHOLD = 0.50
VIRAL_THRESHOLD = 0.25

VIRAL_INDEX = 1
GRADCAM_LAYER = "block5c_project_conv"

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def load_image(image_path, size, channels):
    """Load, resize and normalize an image."""

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

    # RGB classifier preprocessing intentionally remains 0..255.
    return image.numpy()


# ============================================================
# 3. CLASSIFICATION DECISION RULE
# ============================================================

def choose_class(probabilities):
    """
    Apply the previously selected Viral threshold rule.
    """

    viral_probability = float(probabilities[VIRAL_INDEX])

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
# 4. GRAD-CAM
# ============================================================

def build_gradcam_model(classifier):

    backbone = classifier.get_layer("efficientnetb0")
    last_conv_layer = backbone.get_layer(GRADCAM_LAYER)

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output,
        ],
    )

    gradcam_input = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
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


def generate_gradcam(grad_model, rgb_batch, target_class_index):

    image_tensor = tf.convert_to_tensor(
        rgb_batch,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False,
        )

        target_scores = predictions[:, target_class_index]
        target_score_sum = tf.reduce_sum(target_scores)

    gradients = tape.gradient(
        target_score_sum,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable.")

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

    cams = tf.math.divide_no_nan(cams, maximum)

    cams = tf.image.resize(
        cams[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )

    return cams[0, ..., 0].numpy(), predictions[0].numpy()


# ============================================================
# 5. VISUALIZATION HELPERS
# ============================================================

def make_overlay(image_rgb, mask, color):
    """Create a colored mask overlay."""

    image = np.clip(image_rgb / 255.0, 0.0, 1.0)
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


def save_outputs(
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

    # Resize masks only for display.
    infection_display_mask = resize_mask_for_display(
        infection_mask
    )

    lung_overlay = make_overlay(
        image_rgb,
        lung_mask,
        color=(0.0, 1.0, 0.0),
    )

    infection_overlay = make_overlay(
        image_rgb,
        infection_display_mask,
        color=(1.0, 0.0, 0.0),
    )

    # Save input image.
    plt.imsave(
        output_dir / "input_image.png",
        image_display,
    )

    # Save binary masks at their native model resolutions.
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

    # Save individual overlays.
    plt.imsave(
        output_dir / "lung_overlay.png",
        lung_overlay,
    )

    plt.imsave(
        output_dir / "infection_overlay.png",
        infection_overlay,
    )

    # Save Grad-CAM.
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
        f"Grad-CAM: {predicted_class} ({confidence:.2f}%)"
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

    # Save combined visualization.
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
    axes[0, 2].set_title("Infection segmentation (experimental)")

    axes[1, 0].imshow(image_display)
    axes[1, 0].imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(f"Grad-CAM: {predicted_class}")

    axes[1, 1].imshow(
        lung_mask,
        cmap="gray",
    )
    axes[1, 1].set_title("Lung binary mask")

    axes[1, 2].imshow(
        infection_mask,
        cmap="gray",
    )
    axes[1, 2].set_title("Infection binary mask (256x256)")

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
# 6. LOAD ALL MODELS ONCE
# ============================================================

def load_models():

    paths = [
        CLASSIFIER_PATH,
        LUNG_UNET_PATH,
        INFECTION_UNET_PATH,
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}"
            )

    print("\nLoading classification model...")
    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH,
        compile=False,
    )

    print("Loading lung segmentation model...")
    lung_unet = tf.keras.models.load_model(
        LUNG_UNET_PATH,
        compile=False,
    )

    print("Loading infection segmentation model...")
    infection_unet = tf.keras.models.load_model(
        INFECTION_UNET_PATH,
        compile=False,
    )

    print("Building Grad-CAM model...")
    gradcam_model = build_gradcam_model(classifier)

    return classifier, lung_unet, infection_unet, gradcam_model


# ============================================================
# 7. PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    classifier,
    lung_unet,
    infection_unet,
    gradcam_model,
    output_root,
):

    print("\n" + "=" * 65)
    print("Processing:", image_path.name)
    print("=" * 65)

    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # --------------------------------------------------------
    # Lung segmentation
    # --------------------------------------------------------

    print("Running lung segmentation...")

    lung_probabilities = lung_unet(
        gray_batch,
        training=False,
    ).numpy()

    lung_mask = (
        lung_probabilities[0, ..., 0]
        >= LUNG_THRESHOLD
    )

    # --------------------------------------------------------
    # Infection segmentation
    # --------------------------------------------------------

    print("Running infection segmentation...")

    infection_probabilities = infection_unet(
        infection_batch,
        training=False,
    ).numpy()

    infection_mask = (
        infection_probabilities[0, ..., 0]
        >= INFECTION_THRESHOLD
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    print("Running pneumonia classification...")

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

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print("Generating Grad-CAM...")

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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    lung_area_pixels = int(lung_mask.sum())
    infection_area_pixels = int(infection_mask.sum())

    lung_area_percent = float(
        lung_mask.mean() * 100
    )

    infection_area_percent = float(
        infection_mask.mean() * 100
    )

    # Grad-CAM activation inside and outside lung mask.
    lung_activation = None
    background_activation = None

    if lung_mask.any() and (~lung_mask).any() and cam.sum() > 0:

        lung_activation = float(
            cam[lung_mask].sum() / cam.sum() * 100
        )

        background_activation = float(
            cam[~lung_mask].sum() / cam.sum() * 100
        )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    print("Saving output visualizations...")

    save_outputs(
        rgb_image,
        lung_mask,
        infection_mask,
        cam,
        predicted_class,
        confidence,
        output_dir,
    )

    # --------------------------------------------------------
    # JSON result
    # --------------------------------------------------------

    result = {
        "image_name": image_path.name,
        "input_image": str(image_path),

        "predicted_class": predicted_class,
        "selected_class_probability_percent": confidence,
        "decision_rule": decision_rule,

        "class_probabilities_percent": {
            name: float(probabilities[index] * 100)
            for index, name in enumerate(CLASS_NAMES)
        },

        "lung_segmentation": {
            "checkpoint": str(LUNG_UNET_PATH),
            "threshold": LUNG_THRESHOLD,
            "mask_area_pixels": lung_area_pixels,
            "mask_area_percent_of_224_input": lung_area_percent,
            "mask_path": str(output_dir / "lung_mask.png"),
        },

        "infection_segmentation": {
            "checkpoint": str(INFECTION_UNET_PATH),
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

    print("\nClassification:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    print(f"Lung mask area: {lung_area_percent:.2f}%")
    print(f"Infection mask area: {infection_area_percent:.2f}%")
    print("Output:", output_dir)

    return {
        "image_name": image_path.name,
        "predicted_class": predicted_class,
        "confidence_percent": confidence,
        "normal_probability": float(probabilities[0] * 100),
        "viral_probability": float(probabilities[1] * 100),
        "bacterial_probability": float(probabilities[2] * 100),
        "covid_probability": float(probabilities[3] * 100),
        "lung_mask_area_percent": lung_area_percent,
        "infection_mask_area_percent": infection_area_percent,
        "gradcam_inside_lung_percent": lung_activation,
        "status": "success",
        "error": "",
    }


# ============================================================
# 8. INPUT DISCOVERY
# ============================================================

def discover_images(input_path):

    if input_path.is_file():

        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {input_path.suffix}"
            )

        return [input_path]

    if input_path.is_dir():

        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not images:
            raise FileNotFoundError(
                f"No supported images found in: {input_path}"
            )

        return images

    raise FileNotFoundError(
        f"Input path does not exist: {input_path}"
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Complete Pneumonia CDSS inference pipeline."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to one X-ray image or a folder of X-ray images.",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT),
        help="Output directory.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    images = discover_images(input_path)

    print("\n" + "=" * 65)
    print("COMPLETE PNEUMONIA CDSS INFERENCE")
    print("=" * 65)

    print("Input:", input_path)
    print("Images found:", len(images))
    print("Output:", output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load all models only once.
    classifier, lung_unet, infection_unet, gradcam_model = (
        load_models()
    )

    summary_rows = []

    for index, image_path in enumerate(images, start=1):

        print(f"\nImage {index}/{len(images)}")

        try:

            row = process_image(
                image_path,
                classifier,
                lung_unet,
                infection_unet,
                gradcam_model,
                output_root,
            )

            summary_rows.append(row)

        except Exception as error:

            print("ERROR processing:", image_path.name)
            print(error)
            traceback.print_exc()

            summary_rows.append({
                "image_name": image_path.name,
                "predicted_class": "",
                "confidence_percent": None,
                "normal_probability": None,
                "viral_probability": None,
                "bacterial_probability": None,
                "covid_probability": None,
                "lung_mask_area_percent": None,
                "infection_mask_area_percent": None,
                "gradcam_inside_lung_percent": None,
                "status": "failed",
                "error": str(error),
            })

    # Save batch summary.
    summary_df = pd.DataFrame(summary_rows)

    summary_path = output_root / "summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    successful = int(
        (summary_df["status"] == "success").sum()
    )

    failed = len(summary_df) - successful

    print("\n" + "=" * 65)
    print("CDSS INFERENCE COMPLETED")
    print("=" * 65)

    print("Total images:", len(images))
    print("Successful:", successful)
    print("Failed:", failed)
    print("Summary CSV:", summary_path)
    print("Output directory:", output_root)


if __name__ == "__main__":
    main()


# iss ko ek baar run krenge test k liye 
# python .\scripts\116_run_complete_cdss.py --input ".\data\raw\archive(4)\Infection Segmentation Data\Infection Segmentation Data\Test\COVID-19\images\covid_1579.png"


## aagr mood hua toh pua folder b test kr lenge ki result kya and kaisa aa rh hai 
## python .\scripts\116_run_complete_cdss.py --input ".\data\raw\archive(4)\Infection Segmentation Data\Infection Segmentation Data\Test\COVID-19\images"