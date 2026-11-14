"""
SCRIPT 107
Visual Investigation of Infection Segmentation Errors

Selects representative validation cases:
1. Worst overprediction
2. Worst underprediction
3. Worst Dice
4. Representative small infection annotation

Visualizes:
- Original X-ray
- Reference infection mask
- Unconstrained prediction
- Automatic lung-constrained prediction
- False positives
- False negatives

No model training is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

METRICS_CSV = (
    ROOT / "results" / "infection_segmentation"
    / "infection_area_performance_analysis"
    / "infection_area_per_image_analysis.csv"
)

IMAGE_DIR = (
    ROOT / "data" / "raw" / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val" / "COVID-19" / "images"
)

REFERENCE_MASK_DIR = (
    ROOT / "data" / "raw" / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val" / "COVID-19" / "infection masks"
)

PREDICTED_LUNG_MASK_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
    / "predicted_lung_masks"
)

LUNG_MODEL_PATH = (
    ROOT / "saved_models"
    / "unet_lung_segmentation_covid_finetuned_best.keras"
)
INFECTION_MODEL_PATH = (
    ROOT / "saved_models"
    / "unet_covid_infection_segmentation_best.keras"
)
LUNG_THRESHOLD = 0.30
INFECTION_THRESHOLD = 0.30

OUTPUT_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "overprediction_underprediction_visual_investigation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DICE_TOLERANCE = 1e-8


# ============================================================
# FILE LOOKUP
# ============================================================

VALID_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
]


def build_lookup(directory, suffix_to_remove=None):

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )

    lookup = {}

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        stem = path.stem.lower()

        if suffix_to_remove and stem.endswith(suffix_to_remove):
            stem = stem[:-len(suffix_to_remove)]

        lookup[stem] = path

    return lookup


def find_file(image_name, lookup, description):

    stem = Path(str(image_name)).stem.lower()

    if stem not in lookup:
        raise FileNotFoundError(
            f"{description} not found for {image_name}"
        )

    return lookup[stem]


# ============================================================
# LOAD DATA
# ============================================================

def load_grayscale(path):

    return np.asarray(
        Image.open(path).convert("L")
    )


def load_binary_mask(path):

    return load_grayscale(path) >= 128


def resize_mask(mask, shape):

    if mask.shape == shape:
        return mask

    resized = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).resize(
        (shape[1], shape[0]),
        Image.Resampling.NEAREST,
    )

    return np.asarray(resized) >= 128


# ============================================================
# SELECT REPRESENTATIVE CASES
# ============================================================

def select_cases(df):

    selected = []

    # 1. Largest number of false-positive pixels.
    selected.append((
        "WORST_OVERPREDICTION",
        df.loc[df["unconstrained_fp"].idxmax()]
    ))

    # 2. Largest number of false-negative pixels.
    selected.append((
        "WORST_UNDERPREDICTION",
        df.loc[df["unconstrained_fn"].idxmax()]
    ))

    # 3. Lowest unconstrained Dice.
    selected.append((
        "WORST_DICE",
        df.loc[df["unconstrained_dice"].idxmin()]
    ))

    # 4. Representative small infection annotation.
    small_df = df[
        df["reference_infection_area_percentage"].between(
            1, 10, inclusive="both"
        )
    ].copy()

    if not small_df.empty:

        # Choose the median Dice case among small annotations.
        small_df = small_df.sort_values(
            "unconstrained_dice"
        )

        selected.append((
            "REPRESENTATIVE_SMALL_ANNOTATION",
            small_df.iloc[len(small_df) // 2]
        ))

    # Avoid selecting the same image multiple times.
    final_cases = []
    used_names = set()

    for case_type, row in selected:

        image_name = row["image_name"]

        if image_name in used_names:
            continue

        used_names.add(image_name)

        final_cases.append({
            "case_type": case_type,
            "image_name": image_name,
            "reference_area_percentage":
                float(row["reference_infection_area_percentage"]),
            "unconstrained_dice":
                float(row["unconstrained_dice"]),
            "constrained_dice":
                float(row["constrained_dice"]),
            "unconstrained_fp":
                int(row["unconstrained_fp"]),
            "unconstrained_fn":
                int(row["unconstrained_fn"]),
            "unconstrained_area_ratio":
                float(row["unconstrained_area_ratio"]),
        })

    return final_cases


# ============================================================
# VISUALIZATION
# ============================================================

def generate_predictions(image):
    if not LUNG_MODEL_PATH.exists():
        raise FileNotFoundError(LUNG_MODEL_PATH)
    if not INFECTION_MODEL_PATH.exists():
        raise FileNotFoundError(INFECTION_MODEL_PATH)

    # Load trained models
    lung_model = tf.keras.models.load_model(
        LUNG_MODEL_PATH,
        compile=False,
    )
    infection_model = tf.keras.models.load_model(
        INFECTION_MODEL_PATH,
        compile=False,
    )

    # --------------------------------------------------------
    # Lung segmentation: 224 x 224, grayscale, normalized 0-1
    # --------------------------------------------------------
    lung_image = Image.fromarray(
        image.astype(np.uint8)
    ).resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )
    lung_input = (
        np.asarray(lung_image, dtype=np.float32) / 255.0
    )
    lung_input = lung_input[None, ..., None]

    lung_probability = lung_model.predict(
        lung_input,
        verbose=0,
    )[0, ..., 0]
    lung_mask = lung_probability >= LUNG_THRESHOLD

    # --------------------------------------------------------
    # Infection segmentation: 256 x 256, grayscale, 0-1
    # --------------------------------------------------------
    infection_image = Image.fromarray(
        image.astype(np.uint8)
    ).resize(
        (256, 256),
        Image.Resampling.BILINEAR,
    )
    infection_input = (
        np.asarray(infection_image, dtype=np.float32) / 255.0
    )
    infection_input = infection_input[None, ..., None]

    infection_probability = infection_model.predict(
        infection_input,
        verbose=0,
    )[0, ..., 0]
    unconstrained = infection_probability >= INFECTION_THRESHOLD

    # Resize lung mask to match infection prediction size
    lung_mask_resized = resize_mask(
        lung_mask,
        unconstrained.shape,
    )

    # Apply lung constraint
    constrained = unconstrained & lung_mask_resized

    return unconstrained, constrained, lung_mask_resized


def visualize_case(case, image_lookup, reference_lookup):
    image_name = case["image_name"]

    image_path = find_file(
        image_name,
        image_lookup,
        "Original X-ray",
    )

    reference_path = find_file(
        image_name,
        reference_lookup,
        "Reference infection mask",
    )

    image = load_grayscale(image_path)
    reference = load_binary_mask(reference_path)

    if image.shape != reference.shape:
        raise ValueError(
            f"Image/reference shape mismatch: "
            f"{image.shape} vs {reference.shape}"
        )

    # Generate predictions using existing trained models.
    unconstrained, constrained, lung = generate_predictions(image)

    # Resize reference to match prediction size
    reference = resize_mask(
        reference,
        unconstrained.shape,
    )

    # Also resize original image for visualization
    image_vis = np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize(
            (unconstrained.shape[1], unconstrained.shape[0]),
            Image.Resampling.BILINEAR,
        )
    )

    # Error maps
    unconstrained_fp = unconstrained & ~reference
    unconstrained_fn = reference & ~unconstrained

    constrained_fp = constrained & ~reference
    constrained_fn = reference & ~constrained

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))

    fig.suptitle(
        f"{case['case_type']} | {image_name}\n"
        f"Reference area: {case['reference_area_percentage']:.2f}% | "
        f"Unconstrained Dice: {case['unconstrained_dice']:.4f} | "
        f"Constrained Dice: {case['constrained_dice']:.4f}",
        fontsize=13,
    )

    panels = [
        (image_vis, "Original X-ray", "gray"),
        (reference, "Reference infection mask", "gray"),
        (unconstrained, "Unconstrained prediction", "gray"),
        (constrained, "Lung-constrained prediction", "gray"),
        (lung, "Predicted lung mask", "gray"),
        (unconstrained_fp, "Unconstrained false positives", "gray"),
        (unconstrained_fn, "Unconstrained false negatives", "gray"),
        (constrained_fp, "Constrained false positives", "gray"),
        (constrained_fn, "Constrained false negatives", "gray"),
        (np.logical_xor(unconstrained, constrained), "Prediction changes after constraint", "gray"),
    ]

    for axis, (panel, title, cmap) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = (
        OUTPUT_DIR
        / f"{Path(image_name).stem}_{case['case_type'].lower()}.png"
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output_path}")

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INFECTION OVERPREDICTION / UNDERPREDICTION INVESTIGATION")
    print("=" * 80)

    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)

    df = pd.read_csv(METRICS_CSV)

    required = [
        "image_name",
        "reference_infection_area_percentage",
        "unconstrained_dice",
        "constrained_dice",
        "unconstrained_fp",
        "unconstrained_fn",
        "unconstrained_area_ratio",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"\nValidation records: {len(df)}")

    print("\nBuilding image and reference-mask lookups...")
    image_lookup = build_lookup(IMAGE_DIR)
    reference_lookup = build_lookup(REFERENCE_MASK_DIR)

    print(f"Original images discovered: {len(image_lookup)}")
    print(f"Reference masks discovered: {len(reference_lookup)}")

    print("\nChecking trained model files...")
    print(f"Lung model exists: {LUNG_MODEL_PATH.exists()}")
    print(f"Infection model exists: {INFECTION_MODEL_PATH.exists()}")

    if not LUNG_MODEL_PATH.exists():
        raise FileNotFoundError(LUNG_MODEL_PATH)
    if not INFECTION_MODEL_PATH.exists():
        raise FileNotFoundError(INFECTION_MODEL_PATH)

    cases = select_cases(df)

    print("\nSelected representative cases:")

    for case in cases:
        print(
            f"{case['case_type']}: "
            f"{case['image_name']} | "
            f"area={case['reference_area_percentage']:.3f}% | "
            f"Dice={case['unconstrained_dice']:.4f}"
        )

    output_rows = []

    for case in cases:

        path = visualize_case(
            case,
            image_lookup,
            reference_lookup,
        )

        output_rows.append({
            **case,
            "visualization_path": str(path),
        })

    pd.DataFrame(output_rows).to_csv(
        OUTPUT_DIR / "selected_visual_cases.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("SCRIPT 107 COMPLETED")
    print("=" * 80)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated:")
    print("- Representative case PNG visualizations")
    print("- selected_visual_cases.csv")


if __name__ == "__main__":
    main()