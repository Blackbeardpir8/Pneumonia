from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


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
    / "gradcam_error_analysis"
)

IMAGE_SIZE = 224
LUNG_THRESHOLD = 0.30
GRADCAM_THRESHOLD = 0.50

NUM_IMAGES = 6
RANDOM_STATE = 42

CLASS_NAMES = ["Normal", "Viral", "Bacterial", "COVID"]

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD MODELS AND PREDICTIONS
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

print("Predictions loaded:", len(df))



# ============================================================
# 3. PREPARE GRAD-CAM MODEL (ARCHITECTURE-CORRECTED)
# ============================================================

TARGET_LAYER_NAME = "block5c_project_conv"

# Get the actual trained layers from the loaded classifier
augmentation = classifier.get_layer("data_augmentation")
backbone = classifier.get_layer("efficientnetb0")

gap_layer = classifier.get_layer("global_average_pooling")
dropout_layer = classifier.get_layer("classification_dropout")
output_layer = classifier.get_layer("classification_output")

target_layer = backbone.get_layer(TARGET_LAYER_NAME)

print("Backbone:", backbone.name)
print("Grad-CAM target:", target_layer.name)
print("Target output shape:", target_layer.output.shape)

# Intermediate model:
# Input image -> target convolutional layer + final backbone output

backbone_probe = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ],
    name="efficientnet_gradcam_probe"
)


def get_gradcam_outputs(image_batch):
    with tf.GradientTape() as tape:
        # Run the original preprocessing/augmentation layer
        augmented_images = augmentation(
            image_batch,
            training=False
        )
        # Watch the intermediate convolutional feature maps
        # while calculating the classifier output.
        conv_outputs, backbone_features = backbone_probe(
            augmented_images,
            training=False
        )
        class_features = gap_layer(
            backbone_features
        )
        class_features = dropout_layer(
            class_features,
            training=False
        )
        predictions = output_layer(
            class_features
        )
        # Select all four class scores for gradient connectivity check
        class_scores = tf.reduce_sum(predictions)
    gradients = tape.gradient(
        class_scores,
        conv_outputs
    )
    return conv_outputs, predictions, gradients


print("Grad-CAM model construction completed.")

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

    image = tf.cast(image, tf.float32)

    return image


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

    image = tf.cast(image, tf.float32) / 255.0

    return image


# ============================================================
# 5. GENERATE LUNG MASK
# ============================================================

def predict_lung_mask(image_path):

    unet_input = load_unet_image(image_path)

    unet_input = tf.expand_dims(
        unet_input,
        axis=0
    )

    prediction = unet(
        unet_input,
        training=False
    )

    prediction = prediction[0, :, :, 0].numpy()

    binary_mask = (
        prediction >= LUNG_THRESHOLD
    ).astype(np.uint8)

    return prediction, binary_mask


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
        # Ensure the intermediate feature maps are watched
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
            "Grad-CAM gradients are None. "
            "Intermediate feature maps are disconnected."
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

    maximum = tf.reduce_max(cam)

    cam = tf.math.divide_no_nan(
        cam,
        maximum
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
# 7. ATTENTION INSIDE AND OUTSIDE LUNG
# ============================================================

def calculate_attention_statistics(cam, lung_mask):

    activated = cam >= GRADCAM_THRESHOLD

    inside = activated & (lung_mask == 1)
    outside = activated & (lung_mask == 0)

    total_activated = int(np.sum(activated))
    inside_count = int(np.sum(inside))
    outside_count = int(np.sum(outside))

    if total_activated > 0:

        inside_percentage = (
            inside_count / total_activated
        ) * 100

        outside_percentage = (
            outside_count / total_activated
        ) * 100

    else:

        inside_percentage = None
        outside_percentage = None

    return {
        "activated_pixels": total_activated,
        "activated_inside_lung": inside_count,
        "activated_outside_lung": outside_count,
        "activated_inside_percentage": inside_percentage,
        "activated_outside_percentage": outside_percentage,
        "mean_activation_inside_lung": float(
            np.mean(cam[lung_mask == 1])
        ) if np.any(lung_mask == 1) else None,
        "mean_activation_outside_lung": float(
            np.mean(cam[lung_mask == 0])
        ) if np.any(lung_mask == 0) else None,
    }


# ============================================================
# 8. SELECT THREE GROUPS
# ============================================================

groups = {

    "normal_to_covid": {
        "title": "Normal Misclassified as COVID",
        "source": "covid19_radiography",
        "actual": "Normal",
        "predicted": "COVID",
    },

    "viral_to_bacterial": {
        "title": "Viral Misclassified as Bacterial",
        "source": "bacterial_viral_dataset",
        "actual": "Viral",
        "predicted": "Bacterial",
    },

    "correct_viral": {
        "title": "Correctly Classified Viral",
        "source": "bacterial_viral_dataset",
        "actual": "Viral",
        "predicted": "Viral",
    },
}


# ============================================================
# 9. PROCESS EACH GROUP
# ============================================================

all_statistics = []


def process_group(group_name, config):

    print("\n" + "=" * 65)
    print("Processing:", config["title"])
    print("=" * 65)

    selected = df[
        (df["source_dataset"] == config["source"])
        & (df["actual_class"] == config["actual"])
        & (
            df["threshold_predicted_class"]
            == config["predicted"]
        )
    ].copy()

    print("Available images:", len(selected))

    if selected.empty:
        print("No matching images found.")
        return

    selected = selected.sample(
        n=min(NUM_IMAGES, len(selected)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    group_dir = OUTPUT_DIR / group_name

    group_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(12, len(selected) * 4)
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for index, record in selected.iterrows():

        print(
            f"Processing image {index + 1}/{len(selected)}"
        )

        image_path = resolve_image_path(
            record["processed_path"]
        )

        image_tensor = load_classifier_image(
            image_path
        )

        original_image = (
            image_tensor.numpy() / 255.0
        )

        lung_probability, lung_mask = (
            predict_lung_mask(image_path)
        )

        predicted_class_index = CLASS_TO_INDEX[
            config["predicted"]
        ]

        cam, probabilities = generate_gradcam(
            image_tensor,
            predicted_class_index
        )

        attention_stats = calculate_attention_statistics(
            cam,
            lung_mask
        )

        # Create heatmap overlay
        heatmap = plt.get_cmap("jet")(cam)[..., :3]

        overlay = (
            0.55 * original_image
            + 0.45 * heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            1
        )

        # Save individual outputs
        image_stem = image_path.stem

        plt.imsave(
            group_dir / f"{image_stem}_gradcam.png",
            overlay
        )

        plt.imsave(
            group_dir / f"{image_stem}_lung_mask.png",
            lung_mask,
            cmap="gray"
        )

        plt.imsave(
            group_dir / f"{image_stem}_heatmap.png",
            cam,
            cmap="jet"
        )

        # Contact sheet
        axes[index, 0].imshow(
            original_image
        )

        axes[index, 0].set_title(
            "Original X-ray"
        )

        axes[index, 1].imshow(
            overlay
        )

        axes[index, 1].set_title(
            "Grad-CAM Overlay"
        )

        axes[index, 2].imshow(
            lung_mask,
            cmap="gray"
        )

        axes[index, 2].set_title(
            "Predicted Lung Mask"
        )

        for axis in axes[index]:
            axis.axis("off")

        actual_index = CLASS_TO_INDEX[
            record["actual_class"]
        ]

        all_statistics.append({

            "group": group_name,
            "image": image_stem,
            "source_dataset": record["source_dataset"],
            "actual_class": record["actual_class"],
            "threshold_predicted_class": record[
                "threshold_predicted_class"
            ],
            "gradcam_target_class": config["predicted"],

            "actual_class_probability": float(
                probabilities[actual_index]
            ),

            "gradcam_target_probability": float(
                probabilities[predicted_class_index]
            ),

            "lung_area_percentage": float(
                np.mean(lung_mask) * 100
            ),

            **attention_stats
        })

    fig.suptitle(
        config["title"],
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    contact_sheet_path = (
        group_dir / f"{group_name}_gradcam_contact_sheet.png"
    )

    plt.savefig(
        contact_sheet_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    selected.to_csv(
        group_dir / f"{group_name}_selected_images.csv",
        index=False
    )

    print("Saved contact sheet:", contact_sheet_path)


# ============================================================
# 10. RUN ALL THREE GROUPS
# ============================================================

for group_name, config in groups.items():

    process_group(
        group_name,
        config
    )


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

statistics_df = pd.DataFrame(all_statistics)

statistics_df.to_csv(
    OUTPUT_DIR / "gradcam_attention_statistics.csv",
    index=False
)

summary = {
    "classifier": str(CLASSIFIER_PATH),
    "unet": str(UNET_PATH),
    "lung_threshold": LUNG_THRESHOLD,
    "gradcam_threshold": GRADCAM_THRESHOLD,
    "gradcam_target_layer": TARGET_LAYER_NAME,
    "images_processed": len(statistics_df),
    "groups": {
        name: int(
            (statistics_df["group"] == name).sum()
        )
        for name in groups
    }
}

with open(
    OUTPUT_DIR / "gradcam_analysis_summary.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print("\n" + "=" * 65)
print("GRAD-CAM ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("Images processed:", len(statistics_df))
print("Output directory:", OUTPUT_DIR)
print("\nAttention statistics:")
print(statistics_df.to_string(index=False))