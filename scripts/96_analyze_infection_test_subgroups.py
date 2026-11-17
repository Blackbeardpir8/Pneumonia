
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


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

TEST_DIR = DATASET_ROOT / "Test" / "COVID-19"

IMAGE_DIR = TEST_DIR / "images"
INFECTION_MASK_DIR = TEST_DIR / "infection masks"
LUNG_MASK_DIR = TEST_DIR / "lung masks"

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
    / "subgroup_analysis"
)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MASK_THRESHOLD = 128

# Ratios are expressed as percentages.
SMALL_MAX = 10.0
MODERATE_MAX = 50.0


# ============================================================
# 2. FIND MASK FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg"
}


def build_file_index(directory):

    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


# ============================================================
# 3. LOAD AND BINARIZE MASKS
# ============================================================

def load_binary_mask(mask_path):

    mask = tf.io.decode_image(
        tf.io.read_file(str(mask_path)),
        channels=1,
        expand_animations=False,
    )

    mask.set_shape([None, None, 1])

    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method="nearest",
    )

    mask = tf.cast(mask, tf.float32)

    mask = tf.cast(
        mask >= MASK_THRESHOLD,
        tf.uint8,
    )

    return mask.numpy().squeeze()


# ============================================================
# 4. CLASSIFY EACH TEST IMAGE
# ============================================================

def classify_sample(infection_mask, lung_mask):

    infection_pixels = int(
        np.sum(infection_mask)
    )

    lung_pixels = int(
        np.sum(lung_mask)
    )

    if lung_pixels == 0:
        return {
            "group": "INVALID_EMPTY_LUNG_MASK",
            "infection_pixels": infection_pixels,
            "lung_pixels": lung_pixels,
            "infection_lung_ratio_percent": np.nan,
            "masks_identical": False,
        }

    ratio = (
        infection_pixels / lung_pixels
    ) * 100.0

    masks_identical = bool(
        np.array_equal(
            infection_mask,
            lung_mask,
        )
    )

    # Give identical-to-lung masks their own group.
    if masks_identical:

        group = "IDENTICAL_TO_LUNG"

    elif ratio < SMALL_MAX:

        group = "NON_IDENTICAL_SMALL"

    elif ratio <= MODERATE_MAX:

        group = "NON_IDENTICAL_MODERATE"

    else:

        group = "NON_IDENTICAL_LARGE"

    return {
        "group": group,
        "infection_pixels": infection_pixels,
        "lung_pixels": lung_pixels,
        "infection_lung_ratio_percent": float(ratio),
        "masks_identical": masks_identical,
    }


# ============================================================
# 5. SUBGROUP METRICS
# ============================================================

def calculate_group_summary(data):

    records = []

    group_order = [
        "IDENTICAL_TO_LUNG",
        "NON_IDENTICAL_SMALL",
        "NON_IDENTICAL_MODERATE",
        "NON_IDENTICAL_LARGE",
        "INVALID_EMPTY_LUNG_MASK",
    ]

    for group in group_order:

        subset = data[
            data["annotation_group"] == group
        ]

        if subset.empty:
            continue

        records.append({

            "group": group,

            "sample_count": int(len(subset)),

            "mean_dice": float(
                subset["dice"].mean()
            ),

            "median_dice": float(
                subset["dice"].median()
            ),

            "mean_iou": float(
                subset["iou"].mean()
            ),

            "median_iou": float(
                subset["iou"].median()
            ),

            "mean_precision": float(
                subset["precision"].mean()
            ),

            "mean_recall": float(
                subset["recall"].mean()
            ),

            "mean_infection_lung_ratio_percent": float(
                subset[
                    "infection_lung_ratio_percent"
                ].mean()
            ),

            "median_infection_lung_ratio_percent": float(
                subset[
                    "infection_lung_ratio_percent"
                ].median()
            ),

            "identical_mask_count": int(
                subset["masks_identical"].sum()
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# 6. MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("COVID INFECTION SEGMENTATION - TEST SUBGROUP ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)

    for directory in [
        IMAGE_DIR,
        INFECTION_MASK_DIR,
        LUNG_MASK_DIR,
    ]:
        if not directory.exists():
            raise FileNotFoundError(directory)

    # --------------------------------------------------------
    # Load existing test prediction metrics
    # --------------------------------------------------------

    metrics = pd.read_csv(METRICS_PATH)

    print("\nTest prediction records:", len(metrics))

    image_files = build_file_index(IMAGE_DIR)
    infection_files = build_file_index(INFECTION_MASK_DIR)
    lung_files = build_file_index(LUNG_MASK_DIR)

    records = []

    print("\nAnalyzing infection and lung masks...")

    # --------------------------------------------------------
    # Compare reference infection masks with lung masks
    # --------------------------------------------------------

    for index, row in metrics.iterrows():

        image_name = str(row["image"])

        if image_name not in image_files:
            raise FileNotFoundError(
                f"Image missing: {image_name}"
            )

        if image_name not in infection_files:
            raise FileNotFoundError(
                f"Infection mask missing: {image_name}"
            )

        if image_name not in lung_files:
            raise FileNotFoundError(
                f"Lung mask missing: {image_name}"
            )

        infection_mask = load_binary_mask(
            infection_files[image_name]
        )

        lung_mask = load_binary_mask(
            lung_files[image_name]
        )

        annotation = classify_sample(
            infection_mask,
            lung_mask,
        )

        records.append({

            "image": image_name,

            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),

            "annotation_group": annotation["group"],

            "infection_pixels": annotation[
                "infection_pixels"
            ],

            "lung_pixels": annotation[
                "lung_pixels"
            ],

            "infection_lung_ratio_percent": annotation[
                "infection_lung_ratio_percent"
            ],

            "masks_identical": annotation[
                "masks_identical"
            ],
        })

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(metrics)} images"
            )

    data = pd.DataFrame(records)

    # --------------------------------------------------------
    # Verify that every sample was processed
    # --------------------------------------------------------

    if len(data) != len(metrics):
        raise RuntimeError(
            "Some test samples were not analyzed."
        )

    # --------------------------------------------------------
    # Generate subgroup summaries
    # --------------------------------------------------------

    summary = calculate_group_summary(data)

    # --------------------------------------------------------
    # Save per-image results
    # --------------------------------------------------------

    per_image_path = (
        OUTPUT_DIR
        / "test_subgroup_per_image.csv"
    )

    data.to_csv(
        per_image_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save subgroup summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "test_subgroup_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Additional overall annotation statistics
    # --------------------------------------------------------

    identical_count = int(
        data["masks_identical"].sum()
    )

    non_identical_count = int(
        len(data) - identical_count
    )

    valid_data = data[
        data["annotation_group"]
        != "INVALID_EMPTY_LUNG_MASK"
    ]

    overall = {

        "total_test_images": int(len(data)),

        "valid_lung_mask_images": int(
            len(valid_data)
        ),

        "identical_to_lung_count": identical_count,

        "non_identical_count": non_identical_count,

        "identical_to_lung_percentage": float(
            identical_count / len(data) * 100.0
        ),

        "mean_infection_lung_ratio_percent": float(
            valid_data[
                "infection_lung_ratio_percent"
            ].mean()
        ),

        "median_infection_lung_ratio_percent": float(
            valid_data[
                "infection_lung_ratio_percent"
            ].median()
        ),

        "group_counts": {
            str(key): int(value)
            for key, value in
            data["annotation_group"].value_counts().items()
        },

        "interpretation_note": (
            "Subgroups describe the geometry of the supplied "
            "infection annotations relative to lung masks. "
            "They do not establish clinical severity or "
            "annotation correctness."
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "test_subgroup_analysis_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            overall,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANNOTATION QUALITY SUMMARY")
    print("=" * 70)

    print("Total test images:", len(data))

    print(
        "Identical to lung:",
        identical_count,
    )

    print(
        "Non-identical:",
        non_identical_count,
    )

    print(
        "Identical percentage:",
        round(
            overall["identical_to_lung_percentage"],
            2,
        ),
        "%",
    )

    print("\nSUBGROUP PERFORMANCE")
    print("-" * 100)

    if not summary.empty:
        print(
            summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\nSaved files:")
    print(per_image_path)
    print(summary_path)
    print(json_path)

    print("\nSUBGROUP ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/96_analyze_infection_test_subgroups.py