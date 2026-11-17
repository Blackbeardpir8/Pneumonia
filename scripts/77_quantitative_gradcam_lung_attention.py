
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "final_test_evaluation"
    / "original_threshold_0_250_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "quantitative_gradcam_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

NUM_IMAGES_PER_GROUP = 50
RANDOM_STATE = 42

LUNG_THRESHOLD = 0.30
GRADCAM_THRESHOLD = 0.50
VIRAL_THRESHOLD = 0.25

# Abort if reconstructed probabilities differ too much
EQUIVALENCE_TOLERANCE = 1e-5

CLASS_NAMES = [
    "Normal",
    "Viral",
    "Bacterial",
    "COVID"
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

PROBABILITY_COLUMNS = [
    "prob_normal",
    "prob_viral",
    "prob_bacterial",
    "prob_covid"
]


# ============================================================
# 2. LOAD MODELS
# ============================================================

print("\nLoading classifier...")

classifier = tf.keras.models.load_model(
    CLASSIFIER_PATH,
    compile=False
)

print("Loading U-Net...")

unet = tf.keras.models.load_model(
    UNET_PATH,
    compile=False
)

print("Loading existing test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

print("Total test predictions:", len(df))


# ============================================================
# 3. RECONSTRUCT ORIGINAL CLASSIFIER PATH
# ============================================================

augmentation = classifier.get_layer("data_augmentation")

backbone = classifier.get_layer("efficientnetb0")

gap_layer = classifier.get_layer("global_average_pooling")

dropout_layer = classifier.get_layer("classification_dropout")

output_layer = classifier.get_layer("classification_output")

target_layer = backbone.get_layer(
    "block5c_project_conv"
)

print("Grad-CAM target:", target_layer.name)
print("Target feature shape:", target_layer.output.shape)

backbone_probe = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ],
    name="quantitative_gradcam_probe"
)


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def resolve_image_path(path_value):

    path = Path(str(path_value))

    if path.exists():
        return path

    project_path = PROJECT_ROOT / path

    if project_path.exists():
        return project_path

    raise FileNotFoundError(
        f"Image not found: {path_value}"
    )


def load_classifier_image(image_path):

    image = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear"
    )

    return tf.cast(image, tf.float32)


def load_unet_image(image_path):

    image = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear"
    )

    return tf.cast(image, tf.float32) / 255.0


# ============================================================
# 5. GENERATE LUNG MASK
# ============================================================

def predict_lung_mask(image_path):

    image = load_unet_image(image_path)

    image = tf.expand_dims(
        image,
        axis=0
    )

    prediction = unet(
        image,
        training=False
    )

    probability_map = prediction[0, :, :, 0].numpy()

    binary_mask = (
        probability_map >= LUNG_THRESHOLD
    ).astype(np.uint8)

    return probability_map, binary_mask


# ============================================================
# 6. CLASSIFIER PROBABILITY + GRAD-CAM
# ============================================================

def generate_gradcam(image_tensor, target_class_index):

    image_batch = tf.expand_dims(
        image_tensor,
        axis=0
    )

    with tf.GradientTape() as tape:

        augmented_images = augmentation(
            image_batch,
            training=False
        )

        conv_outputs, backbone_features = backbone_probe(
            augmented_images,
            training=False
        )

        tape.watch(conv_outputs)

        pooled_features = gap_layer(
            backbone_features
        )

        dropped_features = dropout_layer(
            pooled_features,
            training=False
        )

        predictions = output_layer(
            dropped_features
        )

        class_score = predictions[:, target_class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients are None."
        )

    conv_outputs = conv_outputs[0]
    gradients = gradients[0]

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    cam = tf.nn.relu(cam)

    cam = tf.math.divide_no_nan(
        cam,
        tf.reduce_max(cam)
    )

    cam = tf.image.resize(
        cam[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear"
    )

    cam = tf.squeeze(cam).numpy()

    probabilities = predictions[0].numpy()

    return cam, probabilities


# ============================================================
# 7. CALCULATE ATTENTION STATISTICS
# ============================================================

def calculate_attention_statistics(cam, lung_mask):

    activated = (
        cam >= GRADCAM_THRESHOLD
    )

    inside = (
        activated
        & (lung_mask == 1)
    )

    outside = (
        activated
        & (lung_mask == 0)
    )

    total_activated = int(
        np.sum(activated)
    )

    inside_count = int(
        np.sum(inside)
    )

    outside_count = int(
        np.sum(outside)
    )

    total_lung_pixels = int(
        np.sum(lung_mask)
    )

    total_non_lung_pixels = int(
        lung_mask.size - total_lung_pixels
    )

    inside_percentage = (
        inside_count / total_activated * 100
        if total_activated > 0
        else None
    )

    outside_percentage = (
        outside_count / total_activated * 100
        if total_activated > 0
        else None
    )

    mean_inside = (
        float(np.mean(cam[lung_mask == 1]))
        if total_lung_pixels > 0
        else None
    )

    mean_outside = (
        float(np.mean(cam[lung_mask == 0]))
        if total_non_lung_pixels > 0
        else None
    )

    return {
        "lung_area_percentage": (
            total_lung_pixels / lung_mask.size * 100
        ),

        "activated_pixels": total_activated,

        "activated_inside_lung": inside_count,

        "activated_outside_lung": outside_count,

        "activated_inside_percentage": inside_percentage,

        "activated_outside_percentage": outside_percentage,

        "mean_activation_inside_lung": mean_inside,

        "mean_activation_outside_lung": mean_outside,
    }


# ============================================================
# 8. DEFINE ANALYSIS GROUPS
# ============================================================

groups = {

    "normal_to_covid": {
        "title": "Normal -> COVID",
        "source": "covid19_radiography",
        "actual": "Normal",
        "predicted": "COVID",
    },

    "viral_to_bacterial": {
        "title": "Viral -> Bacterial",
        "source": "bacterial_viral_dataset",
        "actual": "Viral",
        "predicted": "Bacterial",
    },

    "correct_viral": {
        "title": "Correct Viral",
        "source": "bacterial_viral_dataset",
        "actual": "Viral",
        "predicted": "Viral",
    },
}


# ============================================================
# 9. SELECT 50 IMAGES PER GROUP
# ============================================================

selected_groups = []

for group_name, config in groups.items():

    subset = df[
        (df["source_dataset"] == config["source"])
        & (df["actual_class"] == config["actual"])
        & (
            df["threshold_predicted_class"]
            == config["predicted"]
        )
    ].copy()

    available = len(subset)

    print(
        f"\n{group_name}: {available} available images"
    )

    if available < NUM_IMAGES_PER_GROUP:

        raise ValueError(
            f"{group_name} has only {available} images. "
            f"Cannot select {NUM_IMAGES_PER_GROUP}."
        )

    selected = subset.sample(
        n=NUM_IMAGES_PER_GROUP,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    selected["analysis_group"] = group_name

    selected_groups.append(selected)

selected_df = pd.concat(
    selected_groups,
    ignore_index=True
)

print(
    "\nTotal selected images:",
    len(selected_df)
)

selected_df.to_csv(
    OUTPUT_DIR / "selected_analysis_images.csv",
    index=False
)


# ============================================================
# 10. PROCESS SELECTED IMAGES
# ============================================================

all_results = []

for index, record in selected_df.iterrows():

    print(
        f"\nProcessing image {index + 1}/{len(selected_df)}"
    )

    image_path = resolve_image_path(
        record["processed_path"]
    )

    image_tensor = load_classifier_image(
        image_path
    )

    # Target class is the existing threshold prediction
    target_class_index = CLASS_TO_INDEX[
        record["threshold_predicted_class"]
    ]

    cam, model_probabilities = generate_gradcam(
        image_tensor,
        target_class_index
    )

    # Verify the reconstructed prediction against CSV
    saved_probabilities = record[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=np.float64)

    probability_difference = np.abs(
        model_probabilities - saved_probabilities
    )

    max_probability_difference = float(
        np.max(probability_difference)
    )

    if max_probability_difference > EQUIVALENCE_TOLERANCE:

        raise RuntimeError(
            f"Prediction mismatch for {image_path.name}: "
            f"{max_probability_difference}"
        )

    # Generate lung mask
    _, lung_mask = predict_lung_mask(
        image_path
    )

    # Calculate attention statistics
    attention = calculate_attention_statistics(
        cam,
        lung_mask
    )

    result = {

        "analysis_group": record["analysis_group"],

        "image": image_path.stem,

        "source_dataset": record["source_dataset"],

        "actual_class": record["actual_class"],

        "predicted_class": record[
            "threshold_predicted_class"
        ],

        "target_class_probability": float(
            model_probabilities[target_class_index]
        ),

        "max_probability_difference": (
            max_probability_difference
        ),

        **attention
    }

    all_results.append(result)


# ============================================================
# 11. SAVE PER-IMAGE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_path = (
    OUTPUT_DIR / "per_image_attention_statistics.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 12. GROUP-LEVEL SUMMARY
# ============================================================

metric_columns = [

    "lung_area_percentage",

    "activated_inside_percentage",

    "activated_outside_percentage",

    "mean_activation_inside_lung",

    "mean_activation_outside_lung",

    "activated_pixels",

    "max_probability_difference",
]

summary_rows = []

for group_name, group_data in results_df.groupby(
    "analysis_group"
):

    row = {
        "analysis_group": group_name,
        "image_count": len(group_data),
    }

    for metric in metric_columns:

        values = group_data[metric].dropna()

        row[f"{metric}_mean"] = (
            float(values.mean())
            if len(values) else None
        )

        row[f"{metric}_median"] = (
            float(values.median())
            if len(values) else None
        )

        row[f"{metric}_std"] = (
            float(values.std())
            if len(values) > 1 else 0.0
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_path = (
    OUTPUT_DIR / "group_attention_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 13. SAVE JSON SUMMARY
# ============================================================

json_summary = {

    "experiment": "Quantitative Grad-CAM and Lung Attention Comparison",

    "classifier": str(CLASSIFIER_PATH),

    "unet": str(UNET_PATH),

    "images_per_group": NUM_IMAGES_PER_GROUP,

    "total_images": len(results_df),

    "lung_threshold": LUNG_THRESHOLD,

    "gradcam_threshold": GRADCAM_THRESHOLD,

    "gradcam_target_layer": target_layer.name,

    "viral_threshold": VIRAL_THRESHOLD,

    "prediction_equivalence_tolerance": (
        EQUIVALENCE_TOLERANCE
    ),

    "maximum_probability_difference": float(
        results_df["max_probability_difference"].max()
    ),

    "groups": {}
}

for _, row in summary_df.iterrows():

    group_name = row["analysis_group"]

    json_summary["groups"][group_name] = {

        "image_count": int(row["image_count"]),

        "lung_area_mean": row[
            "lung_area_percentage_mean"
        ],

        "inside_activation_mean": row[
            "activated_inside_percentage_mean"
        ],

        "outside_activation_mean": row[
            "activated_outside_percentage_mean"
        ],

        "mean_activation_inside_lung": row[
            "mean_activation_inside_lung_mean"
        ],

        "mean_activation_outside_lung": row[
            "mean_activation_outside_lung_mean"
        ],
    }

with open(
    OUTPUT_DIR / "quantitative_gradcam_summary.json",
    "w"
) as file:

    json.dump(
        json_summary,
        file,
        indent=4
    )


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("QUANTITATIVE GRAD-CAM AND LUNG ATTENTION ANALYSIS")
print("=" * 80)

print("\nImages processed:", len(results_df))

print(
    "\nMaximum probability difference:",
    results_df["max_probability_difference"].max()
)

print("\nGROUP-WISE RESULTS")

display_columns = [
    "analysis_group",
    "image_count",
    "lung_area_percentage_mean",
    "activated_inside_percentage_mean",
    "activated_outside_percentage_mean",
    "mean_activation_inside_lung_mean",
    "mean_activation_outside_lung_mean",
]

print(
    summary_df[display_columns].to_string(
        index=False
    )
)

print("\nSaved results:")

print(results_path)
print(summary_path)
print(
    OUTPUT_DIR / "quantitative_gradcam_summary.json"
)

print("\nAnalysis completed successfully.")


# # to run this script, use the following command:
# python scripts/77_quantitative_gradcam_lung_attention.py