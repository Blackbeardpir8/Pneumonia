
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = Path.cwd()

MANIFEST_PATH = (
    PROJECT_ROOT / "data/processed/segmentation/manifest.csv"
)

ORIGINAL_MODEL_PATH = (
    PROJECT_ROOT / "saved_models/unet_lung_segmentation.keras"
)

BALANCED_MODEL_PATH = (
    PROJECT_ROOT / "saved_models/unet_lung_segmentation_balanced.keras"
)

ORIGINAL_METRICS_PATH = (
    PROJECT_ROOT / "results/segmentation/validation_metrics.csv"
)

BALANCED_METRICS_PATH = (
    PROJECT_ROOT / "results/segmentation/balanced_validation_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results/segmentation/visual_comparison"
)

THRESHOLD = 0.30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD MODELS
# =====================================================

print("Loading original U-Net...")
original_model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH,
    compile=False
)

print("Loading balanced U-Net...")
balanced_model = tf.keras.models.load_model(
    BALANCED_MODEL_PATH,
    compile=False
)


# =====================================================
# LOAD DATA
# =====================================================

manifest = pd.read_csv(MANIFEST_PATH)

validation = manifest[
    manifest["split"].str.lower() == "val"
].copy()

original_metrics = pd.read_csv(ORIGINAL_METRICS_PATH)
balanced_metrics = pd.read_csv(BALANCED_METRICS_PATH)

comparison = original_metrics[
    ["image", "dice"]
].merge(
    balanced_metrics[["image", "dice"]],
    on="image",
    suffixes=("_original", "_balanced")
)

comparison["improvement"] = (
    comparison["dice_balanced"]
    - comparison["dice_original"]
)

comparison["source"] = (
    comparison["image"].str.split("_").str[0]
)


# =====================================================
# SELECT REPRESENTATIVE IMAGES
# =====================================================

selected = []

for source in ["CHNCXR", "MCUCXR"]:

    source_data = comparison[
        comparison["source"] == source
    ].sort_values("improvement")

    # Lowest improvement
    selected.append(source_data.iloc[0]["image"])

    # Median improvement
    selected.append(
        source_data.iloc[len(source_data) // 2]["image"]
    )

    # Highest improvement
    selected.append(source_data.iloc[-1]["image"])


print("\nSelected validation images:")

for image_name in selected:
    print(image_name)


# =====================================================
# IMAGE LOADING
# =====================================================

def resolve_path(value):

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_grayscale(path):

    image = Image.open(path).convert("L")

    image = np.array(image, dtype=np.float32)

    return image / 255.0


def predict_mask(model, image):

    model_input = image[np.newaxis, ..., np.newaxis]

    prediction = model.predict(
        model_input,
        verbose=0
    )[0, :, :, 0]

    binary_mask = prediction >= THRESHOLD

    return binary_mask.astype(np.uint8), prediction


# =====================================================
# VISUAL COMPARISON
# =====================================================

for image_name in selected:

    row = validation[
        validation["processed_image"].apply(
            lambda p: Path(p).name == image_name
        )
    ]

    if row.empty:
        print("Missing validation image:", image_name)
        continue

    row = row.iloc[0]

    image_path = resolve_path(row["processed_image"])
    mask_path = resolve_path(row["processed_mask"])

    xray = load_grayscale(image_path)

    ground_truth = (
        np.array(
            Image.open(mask_path).convert("L")
        ) > 0
    ).astype(np.uint8)

    original_prediction, _ = predict_mask(
        original_model,
        xray
    )

    balanced_prediction, _ = predict_mask(
        balanced_model,
        xray
    )

    # Find metrics
    metric_row = comparison[
        comparison["image"] == image_name
    ].iloc[0]

    original_dice = metric_row["dice_original"]
    balanced_dice = metric_row["dice_balanced"]

    # =================================================
    # CREATE VISUALIZATION
    # =================================================

    fig, axes = plt.subplots(
        1, 5,
        figsize=(19, 5)
    )

    axes[0].imshow(xray, cmap="gray")
    axes[0].set_title("Original X-ray")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(original_prediction, cmap="gray")
    axes[2].set_title(
        f"Original U-Net\nDice: {original_dice:.4f}"
    )

    axes[3].imshow(balanced_prediction, cmap="gray")
    axes[3].set_title(
        f"Balanced U-Net\nDice: {balanced_dice:.4f}"
    )

    # Overlay: ground truth and balanced prediction
    overlay = np.zeros(
        (*ground_truth.shape, 3),
        dtype=np.float32
    )

    # Green = correct overlap
    # Red = missed ground-truth pixels
    # Blue = extra predicted pixels

    overlap = (
        (ground_truth == 1)
        & (balanced_prediction == 1)
    )

    missed = (
        (ground_truth == 1)
        & (balanced_prediction == 0)
    )

    extra = (
        (ground_truth == 0)
        & (balanced_prediction == 1)
    )

    overlay[overlap] = [0, 1, 0]
    overlay[missed] = [1, 0, 0]
    overlay[extra] = [0, 0, 1]

    axes[4].imshow(overlay)
    axes[4].set_title(
        "Balanced Prediction Overlay\n"
        "Green: overlap | Red: missed | Blue: extra"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_name} | "
        f"Dice Improvement: "
        f"{balanced_dice - original_dice:+.4f}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        image_name.replace(".png", "_comparison.png")
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


print("\nVisual comparison completed.")
print("Output directory:", OUTPUT_DIR)


# to run script, use the following command:
# python scripts/23_visual_compare_segmentation.py