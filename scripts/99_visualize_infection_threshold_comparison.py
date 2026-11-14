
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val"
    / "COVID-19"
)

IMAGE_DIR = DATA_DIR / "images"
INFECTION_MASK_DIR = DATA_DIR / "infection masks"
LUNG_MASK_DIR = DATA_DIR / "lung masks"

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "threshold_visual_comparison"
)

IMAGE_SIZE = 256

THRESHOLDS = [0.30, 0.40, 0.50, 0.70]

SELECTED_THRESHOLD = 0.40

MASK_THRESHOLD = 128

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE MATCHING
# ============================================================

def find_matching_file(folder, image_stem):

    matches = list(folder.glob(image_stem + ".*"))

    if not matches:
        raise FileNotFoundError(
            f"No matching file found for {image_stem} in {folder}"
        )

    return matches[0]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_grayscale_image(path):

    image = Image.open(path).convert("L")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(image, dtype=np.float32)

    image = image / 255.0

    return image


def load_binary_mask(path):

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(mask)

    mask = (mask >= MASK_THRESHOLD).astype(np.uint8)

    return mask


# ============================================================
# METRICS
# ============================================================

def calculate_dice(reference, prediction):

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    intersection = np.logical_and(
        reference,
        prediction
    ).sum()

    total = reference.sum() + prediction.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total


def calculate_area_ratio(infection_mask, lung_mask):

    lung_area = lung_mask.sum()

    if lung_area == 0:
        return np.nan

    return infection_mask.sum() / lung_area


# ============================================================
# DETERMINE ANNOTATION GROUP
# ============================================================

def determine_annotation_group(infection_mask, lung_mask):

    if np.array_equal(infection_mask, lung_mask):

        return "IDENTICAL_TO_LUNG"

    ratio = calculate_area_ratio(
        infection_mask,
        lung_mask
    )

    if np.isnan(ratio):
        return "INVALID"

    if ratio < 0.10:
        return "NON_IDENTICAL_SMALL"

    elif ratio < 0.50:
        return "NON_IDENTICAL_MODERATE"

    else:
        return "NON_IDENTICAL_LARGE"


# ============================================================
# LOAD VALIDATION SAMPLES
# ============================================================

def load_validation_samples():

    samples = []

    image_files = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
    ])

    print(f"Found validation images: {len(image_files)}")

    for index, image_path in enumerate(image_files, start=1):

        stem = image_path.stem

        infection_path = find_matching_file(
            INFECTION_MASK_DIR,
            stem
        )

        lung_path = find_matching_file(
            LUNG_MASK_DIR,
            stem
        )

        image = load_grayscale_image(image_path)

        infection_mask = load_binary_mask(
            infection_path
        )

        lung_mask = load_binary_mask(
            lung_path
        )

        group = determine_annotation_group(
            infection_mask,
            lung_mask
        )

        if group == "INVALID":
            continue

        area_ratio = calculate_area_ratio(
            infection_mask,
            lung_mask
        )

        samples.append({
            "image_name": image_path.name,
            "image": image,
            "infection_mask": infection_mask,
            "lung_mask": lung_mask,
            "annotation_group": group,
            "area_ratio": area_ratio
        })

        if index % 100 == 0:
            print(
                f"Loaded {index}/{len(image_files)} images"
            )

    print(f"\nSuccessfully loaded: {len(samples)}")

    return samples


# ============================================================
# SELECT REPRESENTATIVE SAMPLES
# ============================================================

def select_representative_samples(samples):

    selected = []

    target_groups = [
        "NON_IDENTICAL_SMALL",
        "NON_IDENTICAL_MODERATE",
        "NON_IDENTICAL_LARGE"
    ]

    for group in target_groups:

        group_samples = [
            sample
            for sample in samples
            if sample["annotation_group"] == group
        ]

        if not group_samples:
            print(f"WARNING: No samples found for {group}")
            continue

        # Select the sample nearest to the group's
        # median infection-to-lung area ratio.
        ratios = np.array([
            sample["area_ratio"]
            for sample in group_samples
        ])

        median_ratio = np.median(ratios)

        representative = min(
            group_samples,
            key=lambda sample: abs(
                sample["area_ratio"] - median_ratio
            )
        )

        selected.append(representative)

        print(
            f"{group}: {representative['image_name']} "
            f"(area ratio: "
            f"{representative['area_ratio']:.4f})"
        )

    return selected


# ============================================================
# PREDICTION
# ============================================================

def predict_probability_map(model, image):

    model_input = np.expand_dims(
        image,
        axis=(0, -1)
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )

    return prediction[0, :, :, 0]


# ============================================================
# ERROR MAP
# ============================================================

def create_error_map(reference, prediction):

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    error_map = np.zeros(
        (*reference.shape, 3),
        dtype=np.float32
    )

    # False positives: red
    false_positive = np.logical_and(
        prediction,
        ~reference
    )

    # False negatives: blue
    false_negative = np.logical_and(
        reference,
        ~prediction
    )

    error_map[false_positive] = [1.0, 0.0, 0.0]

    error_map[false_negative] = [0.0, 0.0, 1.0]

    # Correct overlap: green
    true_positive = np.logical_and(
        reference,
        prediction
    )

    error_map[true_positive] = [0.0, 1.0, 0.0]

    return error_map


# ============================================================
# VISUALIZE ONE SAMPLE
# ============================================================

def visualize_sample(model, sample):

    image = sample["image"]

    reference = sample["infection_mask"]

    lung_mask = sample["lung_mask"]

    probability_map = predict_probability_map(
        model,
        image
    )

    predictions = {}

    dice_scores = {}

    for threshold in THRESHOLDS:

        prediction = (
            probability_map >= threshold
        ).astype(np.uint8)

        predictions[threshold] = prediction

        dice_scores[threshold] = calculate_dice(
            reference,
            prediction
        )

    selected_prediction = predictions[
        SELECTED_THRESHOLD
    ]

    error_map = create_error_map(
        reference,
        selected_prediction
    )

    fig, axes = plt.subplots(
        1,
        8,
        figsize=(25, 5)
    )

    axes[0].imshow(
        image,
        cmap="gray"
    )
    axes[0].set_title("Original X-ray")

    axes[1].imshow(
        lung_mask,
        cmap="gray"
    )
    axes[1].set_title("Lung Mask")

    axes[2].imshow(
        reference,
        cmap="gray"
    )
    axes[2].set_title("Reference Infection")

    for index, threshold in enumerate(THRESHOLDS):

        axes[index + 3].imshow(
            image,
            cmap="gray"
        )

        axes[index + 3].imshow(
            predictions[threshold],
            cmap="autumn",
            alpha=0.55
        )

        axes[index + 3].set_title(
            f"Threshold {threshold:.2f}\n"
            f"Dice: {dice_scores[threshold]:.4f}"
        )

    axes[7].imshow(error_map)

    axes[7].set_title(
        "Error Map at 0.40\n"
        "Green: Correct | Red: FP | Blue: FN"
    )

    for axis in axes:

        axis.axis("off")

    fig.suptitle(
        f"{sample['image_name']} | "
        f"{sample['annotation_group']} | "
        f"Infection/Lung Ratio: "
        f"{sample['area_ratio']:.2%}",
        fontsize=13
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{Path(sample['image_name']).stem}_threshold_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved visualization: {output_path}")

    return {
        "image_name": sample["image_name"],
        "annotation_group": sample["annotation_group"],
        "infection_lung_ratio": sample["area_ratio"],
        **{
            f"dice_{threshold:.2f}": dice_scores[threshold]
            for threshold in THRESHOLDS
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INFECTION SEGMENTATION THRESHOLD VISUAL COMPARISON")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nInfection U-Net loaded successfully.")

    samples = load_validation_samples()

    selected_samples = select_representative_samples(
        samples
    )

    if not selected_samples:
        raise RuntimeError(
            "No representative validation samples selected."
        )

    results = []

    for sample in selected_samples:

        print("\n" + "-" * 60)

        print(
            f"Visualizing: {sample['image_name']}"
        )

        result = visualize_sample(
            model,
            sample
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "threshold_visual_comparison.csv"

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("VISUAL COMPARISON COMPLETED")
    print("=" * 70)

    print("\nSelected sample results:")

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved files:")
    print(OUTPUT_DIR)

    print("\nCSV:")
    print(csv_path)


if __name__ == "__main__":
    main()
