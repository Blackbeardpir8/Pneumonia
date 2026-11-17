
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_evaluation"
    / "validation_predictions.csv"
)

SEGMENTATION_TRAIN_DIR = (
    PROJECT_ROOT
    / "data/processed/segmentation/train/images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification"
    / "unet_input_distribution_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# =========================================================
# IMAGE LOADING
# =========================================================

def load_unet_input(image_path):
    """
    Reproduce the U-Net grayscale preprocessing:
    decode grayscale -> resize 224x224 -> normalize 0-1.

    Also retain original dimensions for analysis.
    """

    image_bytes = tf.io.read_file(str(image_path))

    original_image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    original_height = int(original_image.shape[0])
    original_width = int(original_image.shape[1])

    if original_height <= 0 or original_width <= 0:
        raise ValueError(
            f"Invalid image dimensions: {image_path}"
        )

    resized_image = tf.image.resize(
        original_image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    normalized_image = (
        tf.cast(resized_image, tf.float32) / 255.0
    )

    return (
        normalized_image.numpy().squeeze(),
        original_height,
        original_width,
    )


# =========================================================
# CALCULATE IMAGE STATISTICS
# =========================================================

def calculate_image_statistics(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return {
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
        "min_intensity": float(np.min(image)),
        "max_intensity": float(np.max(image)),

        "median_intensity": float(
            np.median(image)
        ),

        "p05_intensity": float(
            np.percentile(image, 5)
        ),

        "p25_intensity": float(
            np.percentile(image, 25)
        ),

        "p75_intensity": float(
            np.percentile(image, 75)
        ),

        "p95_intensity": float(
            np.percentile(image, 95)
        ),

        "dark_pixel_percent": float(
            np.mean(image <= 0.05) * 100
        ),

        "bright_pixel_percent": float(
            np.mean(image >= 0.95) * 100
        ),

        "mid_intensity_pixel_percent": float(
            np.mean(
                (image > 0.05) & (image < 0.95)
            ) * 100
        ),
    }


# =========================================================
# ADD ONE IMAGE TO RESULTS
# =========================================================

def inspect_image(
    image_path,
    dataset_group,
    source_dataset
):

    image_path = Path(image_path)

    image, original_height, original_width = (
        load_unet_input(image_path)
    )

    statistics = calculate_image_statistics(image)

    aspect_ratio = (
        original_width / original_height
    )

    return {
        "dataset_group": dataset_group,
        "source_dataset": source_dataset,
        "image_path": str(image_path),

        "original_height": original_height,
        "original_width": original_width,

        "original_aspect_ratio": aspect_ratio,

        "resized_height": IMAGE_SIZE,
        "resized_width": IMAGE_SIZE,

        **statistics,
    }


# =========================================================
# FIND SEGMENTATION TRAINING IMAGES
# =========================================================

def find_segmentation_images():

    if not SEGMENTATION_TRAIN_DIR.exists():
        raise FileNotFoundError(
            "Segmentation training image directory not found:\n"
            f"{SEGMENTATION_TRAIN_DIR}"
        )

    image_paths = sorted(
        path
        for path in SEGMENTATION_TRAIN_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            "No segmentation training images found."
        )

    return image_paths


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("U-NET INPUT DISTRIBUTION COMPARISON")
    print("=" * 70)

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(VALIDATION_CSV)

    validation_df = pd.read_csv(VALIDATION_CSV)

    required_columns = [
        "processed_path",
        "source_dataset",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in validation_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing validation CSV columns: {missing_columns}"
        )

    all_results = []

    # =====================================================
    # GROUP 1: SEGMENTATION TRAINING DATA
    # =====================================================

    print("\n[1/3] Inspecting segmentation training images...")

    segmentation_paths = find_segmentation_images()

    print(
        "Segmentation training images:",
        len(segmentation_paths)
    )

    for index, image_path in enumerate(
        segmentation_paths,
        start=1
    ):

        try:
            result = inspect_image(
                image_path=image_path,
                dataset_group="segmentation_train",
                source_dataset="lung_segmentation_dataset"
            )

            all_results.append(result)

        except Exception as exc:
            print(
                f"Skipping segmentation image: "
                f"{image_path} | {exc}"
            )

        if index % 100 == 0:
            print(
                f"Segmentation images processed: {index}"
            )

    # =====================================================
    # GROUP 2 AND 3: CLASSIFICATION VALIDATION DATA
    # =====================================================

    print("\n[2/3] Inspecting classification validation images...")

    print("\nSource dataset distribution:")
    print(
        validation_df["source_dataset"]
        .value_counts(dropna=False)
        .to_string()
    )

    for source_name, source_df in validation_df.groupby(
        "source_dataset",
        dropna=False
    ):

        if pd.isna(source_name):
            source_label = "unknown"
        else:
            source_label = str(source_name)

        print(
            f"\nInspecting source: {source_label}"
        )

        for _, row in source_df.iterrows():

            image_path = Path(
                str(row["processed_path"])
            )

            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Validation image missing: {image_path}"
                )

            result = inspect_image(
                image_path=image_path,
                dataset_group="classification_validation",
                source_dataset=source_label
            )

            result["actual_class"] = row.get(
                "actual_class",
                None
            )

            all_results.append(result)

        print(
            f"Images processed: {len(source_df)}"
        )

    # =====================================================
    # SAVE PER-IMAGE RESULTS
    # =====================================================

    print("\n[3/3] Creating reports...")

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        raise RuntimeError(
            "No image statistics were generated."
        )

    per_image_path = (
        OUTPUT_DIR / "unet_input_per_image.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # =====================================================
    # SUMMARY BY DATASET
    # =====================================================

    numeric_columns = [
        "original_height",
        "original_width",
        "original_aspect_ratio",
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
        "median_intensity",
        "p05_intensity",
        "p25_intensity",
        "p75_intensity",
        "p95_intensity",
        "dark_pixel_percent",
        "bright_pixel_percent",
        "mid_intensity_pixel_percent",
    ]

    summary = (
        results_df.groupby(
            ["dataset_group", "source_dataset"],
            dropna=False
        )[numeric_columns]
        .agg(["count", "mean", "std", "min", "max"])
    )

    summary.columns = [
        f"{column}_{statistic}"
        for column, statistic in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        OUTPUT_DIR / "unet_input_dataset_summary.csv",
        index=False
    )

    # =====================================================
    # SUMMARY BY CLASSIFICATION CLASS
    # =====================================================

    classification_df = results_df[
        results_df["dataset_group"]
        == "classification_validation"
    ].copy()

    if "actual_class" in classification_df.columns:

        class_summary = (
            classification_df.groupby(
                ["source_dataset", "actual_class"],
                dropna=False
            )[numeric_columns]
            .mean()
            .reset_index()
        )

        class_summary.to_csv(
            OUTPUT_DIR / "unet_input_class_summary.csv",
            index=False
        )

    # =====================================================
    # IMAGE DIMENSION SUMMARY
    # =====================================================

    dimension_summary = (
        results_df.groupby(
            ["dataset_group", "source_dataset"],
            dropna=False
        )
        .agg(
            image_count=("image_path", "size"),
            unique_dimensions=(
                "original_height",
                lambda values: len(values)
            ),
            mean_original_height=(
                "original_height", "mean"
            ),
            mean_original_width=(
                "original_width", "mean"
            ),
            mean_aspect_ratio=(
                "original_aspect_ratio", "mean"
            ),
        )
        .reset_index()
    )

    dimension_summary.to_csv(
        OUTPUT_DIR / "unet_input_dimension_summary.csv",
        index=False
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nTotal images inspected:", len(results_df))

    print("\nDataset intensity summary:")

    display_columns = [
        "dataset_group",
        "source_dataset",
        "mean_intensity_mean",
        "std_intensity_mean",
        "dark_pixel_percent_mean",
        "bright_pixel_percent_mean",
    ]

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- unet_input_per_image.csv")
    print("- unet_input_dataset_summary.csv")
    print("- unet_input_class_summary.csv")
    print("- unet_input_dimension_summary.csv")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/30_compare_unet_input_distributions.py