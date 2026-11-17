
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
    / "results/classification/validation_evaluation/validation_predictions.csv"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models/unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification/lung_threshold_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

THRESHOLDS = [0.30, 0.40, 0.50]

CLASS_ORDER = ["COVID", "Normal", "Viral", "Bacterial"]


# =========================================================
# LOAD IMAGE
# =========================================================

def load_grayscale_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image.numpy().squeeze()


# =========================================================
# SELECT SAME FOUR SAMPLES
# =========================================================

def select_samples():

    df = pd.read_csv(VALIDATION_CSV)

    selected = []

    for class_name in CLASS_ORDER:

        class_df = df[df["actual_class"] == class_name]

        if class_df.empty:
            print(f"WARNING: No validation image found for {class_name}")
            continue

        selected.append(class_df.iloc[0])

    return selected


# =========================================================
# GENERATE COMPARISON
# =========================================================

def generate_comparison(model, sample, index):

    class_name = sample["actual_class"]

    image_path = Path(sample["processed_path"])

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    original = load_grayscale_image(image_path)

    # U-Net inference
    probabilities = model.predict(
        original[None, ..., None],
        verbose=0
    )[0, ..., 0]

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(16, 16)
    )

    # Original image
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original X-ray")

    # Probability map
    axes[0, 1].imshow(probabilities, cmap="viridis")
    axes[0, 1].set_title("U-Net Probability Map")

    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    summary = []

    for row, threshold in enumerate(THRESHOLDS, start=1):

        mask = probabilities >= threshold

        lung_only = original * mask

        background_only = original * (~mask)

        mask_area = float(np.mean(mask) * 100)

        axes[row, 0].imshow(mask, cmap="gray")
        axes[row, 0].set_title(
            f"Mask | Threshold {threshold:.2f}"
        )

        axes[row, 1].imshow(lung_only, cmap="gray")
        axes[row, 1].set_title("Lung Only")

        axes[row, 2].imshow(background_only, cmap="gray")
        axes[row, 2].set_title("Background Only")

        axes[row, 3].imshow(original, cmap="gray")
        axes[row, 3].imshow(
            mask,
            cmap="Reds",
            alpha=0.35
        )
        axes[row, 3].set_title("Mask Overlay")

        summary.append({
            "class": class_name,
            "threshold": threshold,
            "mask_area_percent": mask_area
        })

        print(
            f"{class_name} | Threshold {threshold:.2f} "
            f"| Mask area: {mask_area:.2f}%"
        )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{class_name} - Lung Mask Threshold Comparison",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        f"threshold_comparison_{index}_{class_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return summary


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading validation samples...")

    samples = select_samples()

    print(f"Selected samples: {len(samples)}")

    print("\nLoading balanced U-Net checkpoint...")

    model = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    all_results = []

    for index, sample in enumerate(samples, start=1):

        print(f"\n[{index}/{len(samples)}] {sample['actual_class']}")

        results = generate_comparison(
            model,
            sample,
            index
        )

        all_results.extend(results)

    summary_df = pd.DataFrame(all_results)

    summary_path = OUTPUT_DIR / "threshold_mask_area_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nThreshold comparison completed.")

    print(f"Images saved in: {OUTPUT_DIR}")

    print(f"Summary saved in: {summary_path}")

    print("\nMask area summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/17_compare_lung_mask_thresholds.py