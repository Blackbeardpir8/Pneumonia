
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    PROJECT_ROOT / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

UNET_PATH = (
    PROJECT_ROOT / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)

INPUT_DIR = (
    PROJECT_ROOT / "results" / "classification"
    / "quantitative_gradcam_analysis"
)

SELECTED_IMAGES_PATH = INPUT_DIR / "selected_analysis_images.csv"

STATISTICS_PATH = INPUT_DIR / "per_image_attention_statistics.csv"

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "classification"
    / "final_gradcam_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

LUNG_THRESHOLD = 0.30
GRADCAM_THRESHOLD = 0.50
EQUIVALENCE_TOLERANCE = 1e-5

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

PROBABILITY_COLUMNS = [
    "prob_normal",
    "prob_viral",
    "prob_bacterial",
    "prob_covid",
]

GROUP_TITLES = {
    "normal_to_covid": "Normal -> COVID",
    "viral_to_bacterial": "Viral -> Bacterial",
    "correct_viral": "Correct Viral",
}

QUANTILES = [0.10, 0.50, 0.90]


# ============================================================
# 2. LOAD MODELS AND EXISTING RESULTS
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

print("Loading existing image selections and statistics...")

selected_df = pd.read_csv(SELECTED_IMAGES_PATH)
stats_df = pd.read_csv(STATISTICS_PATH)

print("Selected images:", len(selected_df))
print("Statistics rows:", len(stats_df))


# ============================================================
# 3. RECONSTRUCT EXACT CLASSIFIER PATH FOR GRAD-CAM
# ============================================================

augmentation = classifier.get_layer("data_augmentation")
backbone = classifier.get_layer("efficientnetb0")
gap_layer = classifier.get_layer("global_average_pooling")
dropout_layer = classifier.get_layer("classification_dropout")
output_layer = classifier.get_layer("classification_output")

target_layer = backbone.get_layer("block5c_project_conv")

backbone_probe = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ],
    name="script79_gradcam_probe"
)

print("Grad-CAM target layer:", target_layer.name)


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

    image = tf.expand_dims(image, axis=0)

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
# 6. GENERATE GRAD-CAM
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
        raise RuntimeError("Grad-CAM gradients are None.")

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
# 7. SELECT REPRESENTATIVE IMAGES
# ============================================================

def select_representative_images():

    merged = selected_df.merge(
        stats_df[
            [
                "image",
                "analysis_group",
                "activated_outside_percentage",
            ]
        ],
        left_on=["processed_path", "analysis_group"],
        right_on=["__unused_path__", "analysis_group"],
        how="left"
    ) if False else None

    # Script 77 stores image stems in its statistics CSV.
    selected_df_copy = selected_df.copy()

    selected_df_copy["image"] = (
        selected_df_copy["processed_path"]
        .apply(lambda x: Path(str(x)).stem)
    )

    merged = selected_df_copy.merge(
        stats_df[
            [
                "image",
                "analysis_group",
                "activated_outside_percentage",
            ]
        ],
        on=["image", "analysis_group"],
        how="inner"
    )

    if len(merged) != len(selected_df):
        raise ValueError(
            "Could not match all selected images with statistics. "
            f"Matched {len(merged)} of {len(selected_df)}."
        )

    chosen = []

    for group_name in GROUP_TITLES:

        group_data = merged[
            merged["analysis_group"] == group_name
        ].copy()

        if group_data.empty:
            raise ValueError(
                f"No images found for {group_name}"
            )

        values = group_data[
            "activated_outside_percentage"
        ]

        for quantile in QUANTILES:

            target_value = values.quantile(quantile)

            candidate_index = (
                values - target_value
            ).abs().idxmin()

            record = group_data.loc[candidate_index].copy()

            record["selected_quantile"] = quantile

            chosen.append(record)

    result = pd.DataFrame(chosen).reset_index(drop=True)

    return result


representative_df = select_representative_images()

representative_df.to_csv(
    OUTPUT_DIR / "representative_images.csv",
    index=False
)

print("\nRepresentative images selected:")
print(
    representative_df[
        [
            "analysis_group",
            "actual_class",
            "threshold_predicted_class",
            "activated_outside_percentage",
            "selected_quantile",
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. VISUALIZATION
# ============================================================

def create_visualization(record, image_number):

    image_path = resolve_image_path(
        record["processed_path"]
    )

    image_tensor = load_classifier_image(image_path)

    target_class = record["threshold_predicted_class"]

    target_class_index = CLASS_TO_INDEX[target_class]

    cam, probabilities = generate_gradcam(
        image_tensor,
        target_class_index
    )

    # Verify prediction probabilities against Script 77.
    saved_probabilities = record[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=np.float64)

    max_difference = float(
        np.max(
            np.abs(probabilities - saved_probabilities)
        )
    )

    if max_difference > EQUIVALENCE_TOLERANCE:

        raise RuntimeError(
            f"Prediction mismatch for {image_path.name}: "
            f"{max_difference}"
        )

    _, lung_mask = predict_lung_mask(image_path)

    # Original grayscale image for visualization.
    display_image = load_unet_image(image_path).numpy().squeeze()

    activated = cam >= GRADCAM_THRESHOLD

    inside = activated & (lung_mask == 1)
    outside = activated & (lung_mask == 0)

    # 0 = no activation, 1 = inside, 2 = outside
    attention_regions = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    attention_regions[inside] = 1
    attention_regions[outside] = 2

    region_cmap = ListedColormap([
        "black",
        "lime",
        "deepskyblue"
    ])

    # Heatmap overlay.
    heatmap = plt.get_cmap("jet")(cam)[..., :3]

    overlay = (
        0.55 * display_image[..., None]
        + 0.45 * heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    # Add lung boundary.
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(21, 5)
    )

    axes[0].imshow(
        display_image,
        cmap="gray"
    )
    axes[0].set_title("Original X-ray")

    axes[1].imshow(
        lung_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Predicted Lung Mask")

    axes[2].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Grad-CAM Heatmap")

    axes[3].imshow(overlay)

    axes[3].contour(
        lung_mask,
        levels=[0.5],
        colors=["white"],
        linewidths=0.8
    )

    axes[3].set_title("Grad-CAM Overlay")

    axes[4].imshow(
        attention_regions,
        cmap=region_cmap,
        vmin=0,
        vmax=2
    )

    axes[4].set_title(
        "Inside / Outside Attention"
    )

    for ax in axes:
        ax.axis("off")

    group_title = GROUP_TITLES[
        record["analysis_group"]
    ]

    probability = probabilities[target_class_index]

    fig.suptitle(
        f"{group_title} | Actual: {record['actual_class']} "
        f"| Predicted: {target_class} "
        f"| Probability: {probability:.3f}\n"
        f"Outside-lung activation: "
        f"{record['activated_outside_percentage']:.2f}%",
        fontsize=13
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.87])

    filename = (
        f"{image_number:02d}_"
        f"{record['analysis_group']}_"
        f"{image_path.stem}.png"
    )

    save_path = OUTPUT_DIR / filename

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "image": image_path.stem,
        "analysis_group": record["analysis_group"],
        "actual_class": record["actual_class"],
        "predicted_class": target_class,
        "target_class_probability": float(probability),
        "activated_inside_percentage": float(
            np.sum(inside) / max(np.sum(activated), 1) * 100
        ),
        "activated_outside_percentage": float(
            np.sum(outside) / max(np.sum(activated), 1) * 100
        ),
        "lung_area_percentage": float(
            np.mean(lung_mask) * 100
        ),
        "max_probability_difference": max_difference,
        "figure_path": str(save_path),
    }


# ============================================================
# 9. PROCESS ALL REPRESENTATIVE IMAGES
# ============================================================

print("\nGenerating Grad-CAM visualizations...")

visualization_results = []

for index, record in representative_df.iterrows():

    print(
        f"Processing {index + 1}/{len(representative_df)}: "
        f"{record['analysis_group']}"
    )

    result = create_visualization(
        record,
        index + 1
    )

    visualization_results.append(result)


# ============================================================
# 10. SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(visualization_results)

results_df.to_csv(
    OUTPUT_DIR / "visualization_summary.csv",
    index=False
)

summary = {
    "experiment": "Final Grad-CAM Visual Comparison",
    "total_images": len(results_df),
    "images_per_group": 3,
    "classifier": str(CLASSIFIER_PATH),
    "unet": str(UNET_PATH),
    "gradcam_target_layer": target_layer.name,
    "lung_threshold": LUNG_THRESHOLD,
    "gradcam_threshold": GRADCAM_THRESHOLD,
    "maximum_probability_difference": float(
        results_df["max_probability_difference"].max()
    ),
    "output_directory": str(OUTPUT_DIR),
    "groups": results_df["analysis_group"].value_counts().to_dict(),
}

with open(
    OUTPUT_DIR / "visualization_summary.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("SCRIPT 79 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nImages visualized:", len(results_df))

print(
    "\nMaximum probability difference:",
    results_df["max_probability_difference"].max()
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved files:")
print("- 9 individual high-resolution PNG figures")
print("- representative_images.csv")
print("- visualization_summary.csv")
print("- visualization_summary.json")

print("\nAnalysis completed.")

# to run this script, use the following command:
# python scripts/79_final_gradcam_visual_comparison.py