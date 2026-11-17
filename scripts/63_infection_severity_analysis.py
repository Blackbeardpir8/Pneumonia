"""
Experimental Infection Area and Severity Analysis.

Pipeline:
1. Balanced U-Net lung segmentation.
2. Original EfficientNetB0 pneumonia classification.
3. Grad-CAM generation for selected class.
4. Threshold Grad-CAM activation map.
5. Restrict candidate activation regions to the lung mask.
6. Calculate candidate infection-area percentage.
7. Assign severity category using research-paper thresholds.

IMPORTANT:
- U-Net predicts lung regions, not pneumonia lesions.
- Grad-CAM represents model-discriminative attention.
- The resulting candidate infection-area percentage is experimental.
- It is NOT a validated measurement of infected lung tissue.
- No model is retrained or modified.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    PROJECT_ROOT / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

UNET_PATH = (
    PROJECT_ROOT / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "results"
    / "classification"
    / "infection_severity_analysis"
)

IMAGE_SIZE = 224

LUNG_THRESHOLD = 0.30
GRADCAM_THRESHOLD = 0.50
VIRAL_THRESHOLD = 0.25

GRADCAM_LAYER = "block5c_project_conv"

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]
VIRAL_INDEX = 1


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_rgb_image(image_path):

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

    # Keep 0-255 for EfficientNetB0.
    return tf.cast(image, tf.float32).numpy()


def load_grayscale_image(image_path):

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

    # U-Net expects grayscale normalized to 0-1.
    return (
        tf.cast(image, tf.float32) / 255.0
    ).numpy()


# ============================================================
# CLASSIFICATION DECISION
# ============================================================

def choose_class(probabilities):

    viral_probability = float(
        probabilities[VIRAL_INDEX]
    )

    if viral_probability >= VIRAL_THRESHOLD:
        return VIRAL_INDEX, "viral_threshold"

    non_viral_indices = [
        i for i in range(len(CLASS_NAMES))
        if i != VIRAL_INDEX
    ]

    selected_index = max(
        non_viral_indices,
        key=lambda index: float(probabilities[index]),
    )

    return selected_index, "non_viral_argmax"


# ============================================================
# GRAD-CAM MODEL
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
        name="severity_gradcam_backbone",
    )

    gradcam_input = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="gradcam_input",
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
    )(
        x,
        training=False,
    )

    predictions = classifier.get_layer(
        "classification_output"
    )(x)

    return tf.keras.Model(
        inputs=gradcam_input,
        outputs=[conv_outputs, predictions],
        name="severity_gradcam_model",
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

    return (
        cams[0, ..., 0].numpy(),
        predictions[0].numpy(),
    )


# ============================================================
# LUNG MASK QUALITY ASSESSMENT
# ============================================================

def assess_lung_mask_quality(lung_mask):
    binary_mask = lung_mask.astype(np.uint8)
    total_pixels = int(binary_mask.size)
    lung_pixels = int(np.sum(binary_mask))
    mask_area_percent = (
        lung_pixels / total_pixels * 100
        if total_pixels > 0 else 0.0
    )

    # Identify connected components.
    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    # Component 0 represents the background.
    component_areas = sorted(
        [
            int(stats[i, cv2.CC_STAT_AREA])
            for i in range(1, component_count)
        ],
        reverse=True,
    )

    foreground_components = len(component_areas)
    largest_component_area = (
        component_areas[0]
        if component_areas else 0
    )
    largest_component_share = (
        largest_component_area / lung_pixels * 100
        if lung_pixels > 0 else 0.0
    )

    flags = []
    if lung_pixels == 0:
        flags.append("empty_lung_mask")
    if mask_area_percent < 10:
        flags.append("very_low_lung_area")
    if mask_area_percent > 55:
        flags.append("suspiciously_high_lung_area")
    if foreground_components == 0:
        flags.append("no_foreground_components")
    if foreground_components > 4:
        flags.append("too_many_components")
    if lung_pixels > 0 and largest_component_share < 40:
        flags.append("fragmented_mask")

    if "empty_lung_mask" in flags:
        quality_assessment = "invalid"
    elif flags:
        quality_assessment = "suspicious"
    else:
        quality_assessment = "acceptable"

    return {
        "lung_area_percent_of_image": float(mask_area_percent),
        "lung_pixels": lung_pixels,
        "connected_component_count": foreground_components,
        "largest_component_area": largest_component_area,
        "largest_component_share_percent": float(largest_component_share),
        "flags": flags,
        "quality_assessment": quality_assessment,
    }


# ============================================================
# CANDIDATE INFECTION REGION
# ============================================================

def calculate_candidate_infection_area(
    cam,
    lung_mask,
):

    # Convert inputs to Boolean masks.
    lung_pixels = lung_mask.astype(bool)

    # Threshold normalized Grad-CAM.
    activation_pixels = cam >= GRADCAM_THRESHOLD

    # Retain only activation pixels inside the lungs.
    candidate_infection_mask = (
        activation_pixels & lung_pixels
    )

    total_lung_pixels = int(
        np.sum(lung_pixels)
    )

    candidate_infected_pixels = int(
        np.sum(candidate_infection_mask)
    )

    if total_lung_pixels == 0:

        return {
            "total_lung_pixels": 0,
            "candidate_infected_pixels": 0,
            "candidate_infection_area_percent": None,
            "calculation_valid": False,
        }, candidate_infection_mask

    percentage = (
        candidate_infected_pixels
        / total_lung_pixels
    ) * 100.0

    result = {
        "total_lung_pixels": total_lung_pixels,
        "candidate_infected_pixels": candidate_infected_pixels,
        "candidate_infection_area_percent": float(
            percentage
        ),
        "calculation_valid": True,
    }

    return result, candidate_infection_mask


# ============================================================
# SEVERITY CLASSIFICATION
# ============================================================

def determine_severity(infection_percentage):

    if infection_percentage is None:
        return "Unavailable"

    # Continuous interpretation of the paper's ranges:
    # 0-25 Mild, >25-50 Moderate,
    # >50-75 Severe, >75-100 Critical.

    if infection_percentage <= 25:
        return "Mild"

    elif infection_percentage <= 50:
        return "Moderate"

    elif infection_percentage <= 75:
        return "Severe"

    return "Critical"


# ============================================================
# SAVE VISUALIZATIONS
# ============================================================

def save_visualizations(
    image_rgb,
    lung_mask,
    cam,
    candidate_infection_mask,
    predicted_class,
    output_dir,
):

    image_display = np.clip(
        image_rgb / 255.0,
        0.0,
        1.0,
    )

    lung_mask = lung_mask.astype(bool)
    candidate_infection_mask = (
        candidate_infection_mask.astype(bool)
    )

    # Save lung mask.
    plt.imsave(
        output_dir / "lung_mask.png",
        lung_mask.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # Save thresholded candidate activation mask.
    plt.imsave(
        output_dir / "candidate_infection_mask.png",
        candidate_infection_mask.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # Grad-CAM overlay.
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image_display)

    ax.imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Grad-CAM: {predicted_class}"
    )

    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_dir / "gradcam_overlay.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Candidate infection overlay.
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image_display)

    overlay = np.zeros(
        (*candidate_infection_mask.shape, 4),
        dtype=np.float32,
    )

    # Red candidate activation regions.
    overlay[..., 0] = 1.0
    overlay[..., 3] = (
        candidate_infection_mask.astype(float)
        * 0.55
    )

    ax.imshow(overlay)

    ax.set_title(
        "Candidate Grad-CAM Regions Inside Lung"
    )

    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_dir / "candidate_infection_overlay.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Combined visualization.
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
    )

    axes[0].imshow(image_display)
    axes[0].set_title("Input X-ray")

    axes[1].imshow(lung_mask, cmap="gray")
    axes[1].set_title("U-Net Lung Mask")

    axes[2].imshow(image_display)
    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Grad-CAM")

    axes[3].imshow(image_display)
    axes[3].imshow(overlay)
    axes[3].set_title("Candidate Infection Regions")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_dir / "combined_severity_visualization.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Experimental infection-area and severity analysis."
        )
    )

    parser.add_argument(
        "image",
        help="Path to chest X-ray image.",
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    ).expanduser()

    if not image_path.is_absolute():
        image_path = (
            Path.cwd() / image_path
        ).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {CLASSIFIER_PATH}"
        )

    if not UNET_PATH.is_file():
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {UNET_PATH}"
        )

    output_dir = (
        OUTPUT_ROOT / image_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading trained models...")

    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH,
        compile=False,
    )

    unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False,
    )

    print("Preprocessing image...")

    rgb_image = load_rgb_image(image_path)
    gray_image = load_grayscale_image(image_path)

    rgb_batch = np.expand_dims(
        rgb_image,
        axis=0,
    ).astype(np.float32)

    gray_batch = np.expand_dims(
        gray_image,
        axis=0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # 1. LUNG SEGMENTATION
    # --------------------------------------------------------

    print("Running U-Net lung segmentation...")

    mask_probabilities = unet(
        gray_batch,
        training=False,
    ).numpy()

    if (
        mask_probabilities.ndim != 4
        or mask_probabilities.shape[-1] < 1
    ):
        raise ValueError(
            f"Unexpected U-Net output: {mask_probabilities.shape}"
        )

    lung_mask = (
        mask_probabilities[0, ..., 0]
        >= LUNG_THRESHOLD
    )

    # --------------------------------------------------------
    # 2. CLASSIFICATION
    # --------------------------------------------------------

    print("Running EfficientNetB0 classification...")

    probabilities = classifier(
        rgb_batch,
        training=False,
    ).numpy()[0]

    if probabilities.shape != (
        len(CLASS_NAMES),
    ):
        raise ValueError(
            f"Unexpected classifier output: {probabilities.shape}"
        )

    predicted_index, decision_rule = choose_class(
        probabilities
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence_percent = float(
        probabilities[predicted_index] * 100
    )

    # --------------------------------------------------------
    # 3. GRAD-CAM
    # --------------------------------------------------------

    print("Generating Grad-CAM...")

    grad_model = build_gradcam_model(
        classifier
    )

    cam, gradcam_probabilities = generate_gradcam(
        grad_model,
        rgb_batch,
        predicted_index,
    )

    if not np.allclose(
        probabilities,
        gradcam_probabilities,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Classifier and Grad-CAM outputs differ."
        )

    # --------------------------------------------------------
    # 4. LUNG MASK QUALITY ASSESSMENT
    # --------------------------------------------------------

    print("Assessing lung segmentation quality...")

    mask_quality = assess_lung_mask_quality(
        lung_mask
    )

    print(
        "Lung area:",
        mask_quality["lung_area_percent_of_image"],
        "%"
    )

    print(
        "Connected components:",
        mask_quality["connected_component_count"]
    )

    print(
        "Quality assessment:",
        mask_quality["quality_assessment"]
    )

    # --------------------------------------------------------
    # 5. CANDIDATE INFECTION AREA
    # --------------------------------------------------------

    print("Calculating candidate infection area...")

    area_result, candidate_infection_mask = (
        calculate_candidate_infection_area(
            cam,
            lung_mask,
        )
    )

    # Reject infection-area calculation if lung mask
    # is invalid or requires manual review.

    if mask_quality["quality_assessment"] != "acceptable":

        infection_percentage = None

        area_result["calculation_valid"] = False

        area_result["candidate_infection_area_percent"] = None

        area_result["rejection_reason"] = (
            "Lung segmentation is suspicious or invalid. "
            "Manual review required."
        )

    else:

        infection_percentage = area_result[
            "candidate_infection_area_percent"
        ]
    # --------------------------------------------------------
    # 6. SEVERITY
    # --------------------------------------------------------

    severity = determine_severity(
        infection_percentage
    )

    # --------------------------------------------------------
    # 7. SAVE OUTPUTS
    # --------------------------------------------------------

    save_visualizations(
        rgb_image,
        lung_mask,
        cam,
        candidate_infection_mask,
        predicted_class,
        output_dir,
    )

    # --------------------------------------------------------
    # 8. RESULT JSON
    # --------------------------------------------------------

    result = {

        "input_image": str(image_path),

        "classification": {
            "predicted_class": predicted_class,
            "selected_probability_percent": (
                confidence_percent
            ),
            "decision_rule": decision_rule,

            "class_probabilities_percent": {
                name: float(
                    probabilities[index] * 100
                )
                for index, name in enumerate(
                    CLASS_NAMES
                )
            },
        },

        "segmentation": {
            "model": "Balanced Lightweight U-Net",
            "lung_threshold": LUNG_THRESHOLD,
            "lung_pixels": area_result["total_lung_pixels"],
            "mask_quality": mask_quality,
        },

        "infection_area_analysis": {
            "method": (
                "Thresholded Grad-CAM intersected "
                "with U-Net lung mask"
            ),

            "gradcam_threshold": GRADCAM_THRESHOLD,

            "candidate_infected_pixels": (
                area_result[
                    "candidate_infected_pixels"
                ]
            ),

            "total_lung_pixels": (
                area_result[
                    "total_lung_pixels"
                ]
            ),

            "candidate_infection_area_percent": (
                infection_percentage
            ),

            "calculation_valid": (
                area_result[
                    "calculation_valid"
                ]
            ),

            "interpretation": (
                "Experimental Grad-CAM candidate activation "
                "area, not validated infected tissue area."
            ),
        },

        "severity_analysis": {
            "severity_level": severity,

            "thresholds": {
                "Mild": "0-25%",
                "Moderate": ">25-50%",
                "Severe": ">50-75%",
                "Critical": ">75-100%",
            },

            "basis": (
                "Research paper severity table, "
                "page 7, interpreted continuously."
            ),

            "clinical_validation": False,
        },

        "interpretation_limitations": [
            "U-Net identifies lung regions, not infection lesions.",
            "Grad-CAM represents model-discriminative attention.",
            "Thresholded Grad-CAM regions are not validated lesion masks.",
            "Candidate infection area must not be interpreted "
            "as a clinical infection measurement.",
            "Severity categories are experimental and not "
            "validated against clinical outcomes.",
            "This prototype is not a standalone clinical diagnosis.",
        ],

        "output_directory": str(output_dir),
    }

    with open(
        output_dir / "severity_analysis_result.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 9. PRINT RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("INFECTION AREA AND SEVERITY ANALYSIS")
    print("=" * 65)

    print(
        "\nPredicted class:",
        predicted_class,
    )

    print(
        "Classification probability:",
        f"{confidence_percent:.2f}%",
    )

    print(
        "Candidate infected pixels:",
        area_result["candidate_infected_pixels"],
    )

    print(
        "Total lung pixels:",
        area_result["total_lung_pixels"],
    )

    
    if infection_percentage is None:

        print("\nCandidate infection area: Unavailable")

        print(
            "Severity: Unavailable "
            "(segmentation requires review)"
        )

        print(
            "Quality flags:",
            mask_quality["flags"]
        )

    else:

        print(
            "\nCandidate infection area:",
            f"{infection_percentage:.2f}%",
        )

        print(
            "Experimental severity category:",
            severity,
        )

    print(
        "Mask quality:",
        mask_quality["quality_assessment"]
    )


    print("\nSaved outputs:")
    print(output_dir)

    print("\nIMPORTANT:")
    print(
        "This is a Grad-CAM-based experimental estimate, "
        "not a validated infection measurement."
    )


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/63_infection_severity_analysis.py <path_to_chest_xray_image