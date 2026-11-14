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

INVALID_CSV = (
    PROJECT_ROOT
    / "results/classification/validation_gradcam_analysis"
    / "invalid_mask_inspection"
    / "invalid_mask_inspection_summary.csv"
)

UNET_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet_lung_segmentation_balanced.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification"
    / "unet_resize_method_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
CONTROL_SAMPLE_SIZE = 20
RANDOM_SEED = 42

FINAL_THRESHOLD = 0.30

COMPARISON_THRESHOLDS = [0.10, 0.20, 0.30]

SOURCE_GROUPS = [
    "covid19_radiography",
    "bacterial_viral_dataset",
]

RESIZE_METHODS = [
    "bilinear",
    "bicubic",
    "area",
]


# =========================================================
# IMAGE LOADING
# =========================================================

def load_original_grayscale(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    image = tf.cast(image, tf.float32)

    return image


def resize_image(image, method):

    resized = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method=method,
        antialias=False,
    )

    resized = resized / 255.0

    return resized.numpy().astype(np.float32)


# =========================================================
# IMAGE STATISTICS
# =========================================================

def calculate_image_statistics(image):

    return {
        "input_mean": float(np.mean(image)),
        "input_std": float(np.std(image)),
        "input_min": float(np.min(image)),
        "input_max": float(np.max(image)),
        "input_median": float(np.median(image)),
    }


# =========================================================
# PREDICTION STATISTICS
# =========================================================

def calculate_prediction_statistics(probability_map):

    result = {
        "probability_mean": float(np.mean(probability_map)),
        "probability_std": float(np.std(probability_map)),
        "probability_max": float(np.max(probability_map)),
        "probability_median": float(np.median(probability_map)),
        "probability_p95": float(
            np.percentile(probability_map, 95)
        ),
    }

    for threshold in COMPARISON_THRESHOLDS:

        mask = probability_map >= threshold

        result[
            f"mask_area_at_{threshold:.2f}_percent"
        ] = float(np.mean(mask) * 100)

    return result


# =========================================================
# SELECT FAILED AND SUCCESSFUL IMAGES
# =========================================================

def select_images():

    validation_df = pd.read_csv(VALIDATION_CSV)
    invalid_df = pd.read_csv(INVALID_CSV)

    validation_df["absolute_path"] = (
        validation_df["processed_path"].apply(
            lambda value: (
                Path(value)
                if Path(value).is_absolute()
                else PROJECT_ROOT / Path(value)
            )
        )
    )

    invalid_paths = set(
        invalid_df["processed_path"].apply(
            lambda value: str(
                Path(value)
                if Path(value).is_absolute()
                else PROJECT_ROOT / Path(value)
            )
        )
    )

    failed_df = validation_df[
        validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    if len(failed_df) != len(invalid_paths):
        raise RuntimeError(
            "Failed image matching count does not match "
            "the invalid inspection records."
        )

    if failed_df.empty:
        raise RuntimeError("No failed images were matched.")

    failed_df["comparison_group"] = "failed"

    control_candidates = validation_df[
        ~validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    controls = []

    for source in SOURCE_GROUPS:

        source_df = control_candidates[
            control_candidates["source_dataset"].astype(str)
            == source
        ]

        if source_df.empty:
            raise RuntimeError(
                f"No successful controls found for {source}"
            )

        sample_count = min(
            CONTROL_SAMPLE_SIZE,
            len(source_df),
        )

        selected = source_df.sample(
            n=sample_count,
            random_state=RANDOM_SEED,
        ).copy()

        selected["comparison_group"] = (
            f"successful_{source}"
        )

        controls.append(selected)

    selected_df = pd.concat(
        [failed_df] + controls,
        ignore_index=True,
    )

    return selected_df


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("U-NET RESIZE METHOD COMPARISON")
    print("=" * 70)

    for path in [
        VALIDATION_CSV,
        INVALID_CSV,
        UNET_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    selected_df = select_images()

    print("\nSelected images:")
    print("Failed:", len(
        selected_df[
            selected_df["comparison_group"] == "failed"
        ]
    ))

    print(
        "Successful controls:",
        len(selected_df) -
        len(
            selected_df[
                selected_df["comparison_group"] == "failed"
            ]
        ),
    )

    print("\nLoading balanced U-Net...")

    model = tf.keras.models.load_model(
        UNET_PATH,
        compile=False,
    )

    results = []

    total_images = len(selected_df)

    # =====================================================
    # PROCESS EACH IMAGE USING EACH RESIZE METHOD
    # =====================================================

    for image_number, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1,
    ):

        image_path = Path(row["absolute_path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        original_image = load_original_grayscale(
            image_path
        )

        original_height = int(original_image.shape[0])
        original_width = int(original_image.shape[1])

        aspect_ratio = (
            original_width / original_height
        )

        for method in RESIZE_METHODS:

            input_image = resize_image(
                original_image,
                method,
            )

            probability_map = model(
                input_image[None, ...],
                training=False,
            ).numpy()[0, ..., 0]

            image_stats = calculate_image_statistics(
                input_image
            )

            prediction_stats = (
                calculate_prediction_statistics(
                    probability_map
                )
            )

            result = {
                "comparison_group": row["comparison_group"],
                "source_dataset": row["source_dataset"],
                "actual_class": row.get("actual_class"),
                "predicted_class": row.get("predicted_class"),
                "resize_method": method,
                "original_width": original_width,
                "original_height": original_height,
                "original_aspect_ratio": aspect_ratio,
                "processed_path": str(image_path),
                **image_stats,
                **prediction_stats,
            }

            results.append(result)

        print(
            f"Processed image {image_number}/{total_images}"
        )

    # =====================================================
    # SAVE PER-IMAGE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "resize_method_per_image.csv",
        index=False,
    )

    # =====================================================
    # SUMMARY BY GROUP AND METHOD
    # =====================================================

    metric_columns = [
        "input_mean",
        "input_std",
        "probability_mean",
        "probability_max",
        "probability_p95",
        "mask_area_at_0.10_percent",
        "mask_area_at_0.20_percent",
        "mask_area_at_0.30_percent",
    ]

    summary = (
        results_df.groupby(
            ["comparison_group", "resize_method"],
            dropna=False,
        )
        .agg(
            image_count=("processed_path", "nunique"),
            **{
                f"mean_{column}": (column, "mean")
                for column in metric_columns
            },
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "resize_method_summary.csv",
        index=False,
    )

    # =====================================================
    # COMPARE EACH METHOD AGAINST BILINEAR
    # =====================================================

    baseline = results_df[
        results_df["resize_method"] == "bilinear"
    ].copy()

    comparison_metrics = [
        "probability_mean",
        "probability_max",
        "mask_area_at_0.30_percent",
    ]

    for method in ["bicubic", "area"]:

        alternative = results_df[
            results_df["resize_method"] == method
        ].copy()

        merged = baseline.merge(
            alternative,
            on="processed_path",
            suffixes=("_bilinear", f"_{method}"),
        )

        for metric in comparison_metrics:

            merged[f"{metric}_difference"] = (
                merged[f"{metric}_{method}"]
                - merged[f"{metric}_bilinear"]
            )

        merged.to_csv(
            OUTPUT_DIR / f"{method}_vs_bilinear.csv",
            index=False,
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print("RESIZE METHOD COMPARISON COMPLETED")
    print("=" * 70)

    display_columns = [
        "comparison_group",
        "resize_method",
        "image_count",
        "mean_probability_mean",
        "mean_probability_max",
        "mean_mask_area_at_0.30_percent",
    ]

    print("\nSummary:")
    print(
        summary[display_columns].to_string(index=False)
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- resize_method_per_image.csv")
    print("- resize_method_summary.csv")
    print("- bicubic_vs_bilinear.csv")
    print("- area_vs_bilinear.csv")


if __name__ == "__main__":
    main()