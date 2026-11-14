
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage


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
    / "results/classification/connected_component_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
THRESHOLD = 0.30

CLASS_ORDER = ["COVID", "Normal", "Viral", "Bacterial"]


# =========================================================
# IMAGE LOADING
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
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image.numpy().squeeze()


# =========================================================
# CONNECTED COMPONENT FILTERING
# =========================================================

def keep_largest_two_components(mask):

    structure = np.ones((3, 3), dtype=np.uint8)

    labeled_mask, number_of_components = ndimage.label(
        mask,
        structure=structure
    )

    if number_of_components == 0:
        return np.zeros_like(mask, dtype=bool), 0

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )

    component_sizes[0] = 0

    largest_labels = np.argsort(component_sizes)[-2:]

    filtered_mask = np.isin(
        labeled_mask,
        largest_labels
    )

    filtered_count = int(np.count_nonzero(largest_labels))

    return filtered_mask, filtered_count


# =========================================================
# SAMPLE SELECTION
# =========================================================

def select_samples():

    df = pd.read_csv(VALIDATION_CSV)

    selected = []

    for class_name in CLASS_ORDER:

        class_df = df[df["actual_class"] == class_name]

        if class_df.empty:
            print(f"WARNING: No sample found for {class_name}")
            continue

        selected.append(class_df.iloc[0])

    return selected


# =========================================================
# VISUAL COMPARISON
# =========================================================

def generate_comparison(model, sample, index):

    class_name = sample["actual_class"]

    image_path = Path(sample["processed_path"])

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    original = load_grayscale_image(image_path)

    probabilities = model.predict(
        original[None, ..., None],
        verbose=0
    )[0, ..., 0]

    original_mask = probabilities >= THRESHOLD

    filtered_mask, component_count = keep_largest_two_components(
        original_mask
    )

    original_lung = original * original_mask
    filtered_lung = original * filtered_mask

    original_background = original * (~original_mask)
    filtered_background = original * (~filtered_mask)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 9)
    )

    images = [
        (original, "Original X-ray", "gray"),
        (original_mask, "Original U-Net Mask", "gray"),
        (filtered_mask, "Two Largest Components", "gray"),
        (probabilities, "U-Net Probability Map", "viridis"),
        (original_lung, "Original Lung Only", "gray"),
        (filtered_lung, "Filtered Lung Only", "gray"),
        (original_background, "Original Background Only", "gray"),
        (filtered_background, "Filtered Background Only", "gray"),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, images):

        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{class_name} - Connected Component Comparison",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        f"component_comparison_{index}_{class_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    original_area = float(np.mean(original_mask) * 100)
    filtered_area = float(np.mean(filtered_mask) * 100)

    print(f"\n{class_name}")
    print(f"Connected components detected: {component_count}")
    print(f"Original mask area: {original_area:.2f}%")
    print(f"Filtered mask area: {filtered_area:.2f}%")

    return {
        "class": class_name,
        "threshold": THRESHOLD,
        "components_detected": component_count,
        "original_mask_area_percent": original_area,
        "filtered_mask_area_percent": filtered_area,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("Selecting validation samples...")

    samples = select_samples()

    print(f"Selected samples: {len(samples)}")

    print("\nLoading balanced U-Net...")

    model = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    results = []

    for index, sample in enumerate(samples, start=1):

        print(f"\n[{index}/{len(samples)}] Processing {sample['actual_class']}")

        result = generate_comparison(
            model,
            sample,
            index
        )

        results.append(result)

    summary = pd.DataFrame(results)

    summary_path = OUTPUT_DIR / "component_comparison_summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nComparison completed.")

    print(f"Images: {OUTPUT_DIR}")
    print(f"Summary: {summary_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
