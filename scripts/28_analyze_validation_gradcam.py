
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


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

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification"
    / "validation_gradcam_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

LUNG_THRESHOLD = 0.30
GRADCAM_LAYER = "block5c_project_conv"

CLASS_NAMES = [
    "Normal",
    "Viral",
    "Bacterial",
    "COVID",
]


# =========================================================
# LOAD IMAGE
# =========================================================

def load_rgb_image(image_path):
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

    # EfficientNet input remains in the 0-255 range.
    image = tf.cast(image, tf.float32)

    return image.numpy()


def load_grayscale_image(image_path):
    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    # U-Net input is normalized to 0-1.
    image = tf.cast(image, tf.float32) / 255.0

    return image.numpy()


# =========================================================
# BUILD GRAD-CAM MODEL
# =========================================================

def build_gradcam_model(classifier):

    backbone = classifier.get_layer("efficientnetb0")

    last_conv_layer = backbone.get_layer(
        GRADCAM_LAYER
    )

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ],
        name="validation_gradcam_backbone"
    )

    gradcam_input = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    conv_outputs, backbone_output = backbone_grad_model(
        gradcam_input,
        training=False
    )

    x = classifier.get_layer("global_average_pooling")(
        backbone_output
    )

    x = classifier.get_layer("classification_dropout")(
        x,
        training=False
    )

    predictions = classifier.get_layer(
        "classification_output"
    )(x)

    grad_model = tf.keras.Model(
        inputs=gradcam_input,
        outputs=[conv_outputs, predictions],
        name="validation_gradcam_model"
    )

    print("Grad-CAM layer:", GRADCAM_LAYER)
    print("Feature map:", last_conv_layer.output.shape)

    return grad_model


# =========================================================
# GENERATE BATCH GRAD-CAM
# =========================================================

def generate_batch_gradcam(grad_model, image_batch):

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        predicted_indices = tf.argmax(
            predictions,
            axis=1,
            output_type=tf.int32
        )

        target_scores = tf.gather(
            predictions,
            predicted_indices,
            axis=1,
            batch_dims=1
        )

        target_score_sum = tf.reduce_sum(target_scores)

    gradients = tape.gradient(
        target_score_sum,
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

    cams = tf.reduce_sum(
        conv_outputs
        * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    cams = tf.nn.relu(cams)

    maximum = tf.reduce_max(
        cams,
        axis=(1, 2),
        keepdims=True
    )

    cams = tf.math.divide_no_nan(
        cams,
        maximum
    )

    cams = tf.image.resize(
        cams[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    return (
        cams[..., 0].numpy(),
        predictions.numpy(),
        predicted_indices.numpy()
    )


# =========================================================
# CALCULATE ACTIVATION INSIDE / OUTSIDE LUNGS
# =========================================================

def calculate_attention_metrics(cam, lung_mask):

    lung_pixels = lung_mask > 0
    background_pixels = ~lung_pixels

    lung_area = int(np.sum(lung_pixels))
    total_pixels = lung_mask.size
    background_area = total_pixels - lung_area

    result = {
        "lung_mask_area_percent": (
            lung_area / total_pixels * 100
        ),
        "lung_activation_percent": np.nan,
        "background_activation_percent": np.nan,
        "mean_lung_activation": np.nan,
        "mean_background_activation": np.nan,
        "attention_analysis_valid": False,
    }

    # A zero-area mask cannot support this analysis.
    if lung_area == 0 or background_area == 0:
        return result

    total_activation = float(np.sum(cam))

    if total_activation <= 0:
        return result

    lung_activation = float(
        np.sum(cam[lung_pixels])
    )

    background_activation = float(
        np.sum(cam[background_pixels])
    )

    result.update({
        "lung_activation_percent": (
            lung_activation / total_activation * 100
        ),
        "background_activation_percent": (
            background_activation / total_activation * 100
        ),
        "mean_lung_activation": float(
            np.mean(cam[lung_pixels])
        ),
        "mean_background_activation": float(
            np.mean(cam[background_pixels])
        ),
        "attention_analysis_valid": True,
    })

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("VALIDATION-WIDE GRAD-CAM ATTENTION ANALYSIS")
    print("=" * 65)

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(VALIDATION_CSV)

    df = pd.read_csv(VALIDATION_CSV)

    required_columns = [
        "actual_class",
        "predicted_class",
        "processed_path",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    if "source_dataset" not in df.columns:
        df["source_dataset"] = "unknown"

    df = df.reset_index(drop=True)

    print("\nValidation records:", len(df))
    print("\nLoading classifier...")

    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH,
        compile=False
    )

    print("Loading balanced U-Net...")

    unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    grad_model = build_gradcam_model(classifier)

    results = []

    total_batches = (
        len(df) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("\nStarting analysis...")

    for batch_number, start in enumerate(
        range(0, len(df), BATCH_SIZE),
        start=1
    ):

        batch_df = df.iloc[
            start:start + BATCH_SIZE
        ]

        rgb_images = []
        gray_images = []
        valid_rows = []

        for row_index, row in batch_df.iterrows():

            image_path = Path(
                str(row["processed_path"])
            )

            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            rgb_images.append(
                load_rgb_image(image_path)
            )

            gray_images.append(
                load_grayscale_image(image_path)
            )

            valid_rows.append(row)

        rgb_batch = np.stack(
            rgb_images
        ).astype(np.float32)

        gray_batch = np.stack(
            gray_images
        ).astype(np.float32)

        # Generate Grad-CAM for each image's predicted class.
        cams, predictions, predicted_indices = (
            generate_batch_gradcam(
                grad_model,
                rgb_batch
            )
        )

        # Generate lung masks using the balanced U-Net.
        mask_probabilities = unet(
            gray_batch,
            training=False
        ).numpy()

        lung_masks = (
            mask_probabilities[..., 0]
            >= LUNG_THRESHOLD
        )

        for i, row in enumerate(valid_rows):

            actual_class = str(
                row["actual_class"]
            )

            predicted_class = CLASS_NAMES[
                int(predicted_indices[i])
            ]

            confidence = float(
                predictions[i, predicted_indices[i]] * 100
            )

            attention = calculate_attention_metrics(
                cams[i],
                lung_masks[i]
            )

            result = {
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "source_dataset": row["source_dataset"],
                "prediction_correct": (
                    actual_class == predicted_class
                ),
                "confidence_percent": confidence,
                "processed_path": row["processed_path"],
                **attention,
            }

            results.append(result)

        print(
            f"Processed batch {batch_number}/{total_batches}"
        )

    # =====================================================
    # SAVE PER-IMAGE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    per_image_path = (
        OUTPUT_DIR / "validation_gradcam_per_image.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # =====================================================
    # SUMMARY BY CLASS AND SOURCE
    # =====================================================

    valid_df = results_df[
        results_df["attention_analysis_valid"]
    ].copy()

    metric_columns = [
        "lung_mask_area_percent",
        "lung_activation_percent",
        "background_activation_percent",
        "mean_lung_activation",
        "mean_background_activation",
    ]

    if not valid_df.empty:

        summary_by_class_source = (
            valid_df.groupby(
                ["actual_class", "source_dataset"],
                dropna=False
            )
            .agg(
                image_count=("actual_class", "size"),
                mean_confidence=("confidence_percent", "mean"),
                mean_lung_mask_area=(
                    "lung_mask_area_percent", "mean"
                ),
                mean_lung_activation=(
                    "lung_activation_percent", "mean"
                ),
                mean_background_activation=(
                    "background_activation_percent", "mean"
                ),
                mean_lung_activation_intensity=(
                    "mean_lung_activation", "mean"
                ),
                mean_background_activation_intensity=(
                    "mean_background_activation", "mean"
                ),
            )
            .reset_index()
        )

        summary_by_class = (
            valid_df.groupby("actual_class")
            .agg(
                image_count=("actual_class", "size"),
                mean_lung_mask_area=(
                    "lung_mask_area_percent", "mean"
                ),
                mean_lung_activation=(
                    "lung_activation_percent", "mean"
                ),
                mean_background_activation=(
                    "background_activation_percent", "mean"
                ),
                mean_lung_activation_intensity=(
                    "mean_lung_activation", "mean"
                ),
                mean_background_activation_intensity=(
                    "mean_background_activation", "mean"
                ),
            )
            .reset_index()
        )

        summary_by_class_source.to_csv(
            OUTPUT_DIR / "summary_by_class_and_source.csv",
            index=False
        )

        summary_by_class.to_csv(
            OUTPUT_DIR / "summary_by_class.csv",
            index=False
        )

    # =====================================================
    # DATA QUALITY SUMMARY
    # =====================================================

    quality_summary = {
        "total_validation_images": len(results_df),
        "valid_attention_analyses": int(
            results_df["attention_analysis_valid"].sum()
        ),
        "invalid_attention_analyses": int(
            (~results_df["attention_analysis_valid"]).sum()
        ),
        "zero_or_full_lung_masks": int(
            (
                (results_df["lung_mask_area_percent"] == 0)
                | (results_df["lung_mask_area_percent"] == 100)
            ).sum()
        ),
        "correct_predictions": int(
            results_df["prediction_correct"].sum()
        ),
        "incorrect_predictions": int(
            (~results_df["prediction_correct"]).sum()
        ),
        "lung_threshold": LUNG_THRESHOLD,
        "gradcam_layer": GRADCAM_LAYER,
    }

    pd.DataFrame(
        [quality_summary]
    ).to_csv(
        OUTPUT_DIR / "analysis_quality_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nQuality summary:")
    for key, value in quality_summary.items():
        print(f"{key}: {value}")

    print("\nResults folder:")
    print(OUTPUT_DIR)

    print("\nPer-class summary:")
    if not valid_df.empty:
        print(
            summary_by_class.to_string(index=False)
        )


if __name__ == "__main__":
    main()
