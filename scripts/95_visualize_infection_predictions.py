
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "test_evaluation"
    / "test_image_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "visualizations"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
THRESHOLD = 0.5


# ============================================================
# 2. LOAD IMAGE AND MASK
# ============================================================

def load_image_and_mask(image_path, mask_path):

    image = tf.io.decode_image(
        tf.io.read_file(str(image_path)),
        channels=1,
        expand_animations=False,
    )

    mask = tf.io.decode_image(
        tf.io.read_file(str(mask_path)),
        channels=1,
        expand_animations=False,
    )

    image.set_shape([None, None, 1])
    mask.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
    )

    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = tf.cast(mask >= 128.0, tf.uint8)

    return image.numpy().squeeze(), mask.numpy().squeeze()


# ============================================================
# 3. GENERATE PREDICTION
# ============================================================

def predict_infection(model, image):

    input_image = np.expand_dims(
        image,
        axis=(0, -1),
    )

    probability = model(
        input_image,
        training=False,
    ).numpy()[0, :, :, 0]

    predicted_mask = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    return probability, predicted_mask


# ============================================================
# 4. CREATE VISUAL COMPARISON
# ============================================================

def save_visualization(
    image,
    ground_truth,
    predicted_mask,
    probability,
    image_name,
    dice_score,
    category,
):

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )
    axes[0].set_title("Original Chest X-ray")

    axes[1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Reference Infection Mask")

    axes[2].imshow(
        predicted_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Predicted Infection Mask")

    axes[3].imshow(
        image,
        cmap="gray",
    )

    # Green = reference infection region
    # Red = predicted infection region
    # Yellow = overlap
    overlay = np.zeros(
        (*image.shape, 3),
        dtype=np.float32,
    )

    overlay[:, :, 0] = predicted_mask
    overlay[:, :, 1] = ground_truth

    axes[3].imshow(
        overlay,
        alpha=0.55,
    )

    axes[3].set_title(
        "Overlay: Red=Prediction, Green=Reference"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{category} | {image_name} | Dice: {dice_score:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{category.lower()}_{image_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COVID INFECTION SEGMENTATION - VISUAL ANALYSIS")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    metrics = pd.read_csv(METRICS_PATH)

    if metrics.empty:
        raise ValueError("Test metrics file is empty.")

    # Select best, median and worst Dice cases.
    sorted_metrics = metrics.sort_values(
        "dice"
    ).reset_index(drop=True)

    selected_indices = {
        "WORST": 0,
        "MEDIAN": len(sorted_metrics) // 2,
        "BEST": len(sorted_metrics) - 1,
    }

    print("\nSelected representative images:")

    for category, index in selected_indices.items():

        row = sorted_metrics.iloc[index]

        print(
            f"{category}: {row['image']} "
            f"| Dice={row['dice']:.4f}"
        )

    print("\nLoading saved infection U-Net...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    test_image_dir = (
        DATASET_ROOT / "Test" / "COVID-19" / "images"
    )

    test_mask_dir = (
        DATASET_ROOT / "Test" / "COVID-19" / "infection masks"
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # VISUALIZE SELECTED CASES
    # --------------------------------------------------------

    for category, index in selected_indices.items():

        row = sorted_metrics.iloc[index]

        image_name = str(row["image"])

        image_path = (
            test_image_dir / f"{image_name}.png"
        )

        mask_path = (
            test_mask_dir / f"{image_name}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image, ground_truth = load_image_and_mask(
            image_path,
            mask_path,
        )

        probability, predicted_mask = predict_infection(
            model,
            image,
        )

        save_visualization(
            image=image,
            ground_truth=ground_truth,
            predicted_mask=predicted_mask,
            probability=probability,
            image_name=image_name,
            dice_score=float(row["dice"]),
            category=category,
        )

    print("\n" + "=" * 65)
    print("VISUALIZATION COMPLETED")
    print("=" * 65)

    print("Output directory:", OUTPUT_DIR)
    print("Total images visualized:", len(selected_indices))


if __name__ == "__main__":
    main()

