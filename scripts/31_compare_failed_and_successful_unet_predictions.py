
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
    / "failed_successful_unet_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
CONTROL_SAMPLE_SIZE = 20
RANDOM_SEED = 42

FINAL_THRESHOLD = 0.30

COMPARISON_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
]

SOURCE_GROUPS = [
    "covid19_radiography",
    "bacterial_viral_dataset",
]


# =========================================================
# IMAGE PREPROCESSING
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

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image.numpy()


# =========================================================
# IMAGE STATISTICS
# =========================================================

def calculate_input_statistics(image):

    image = np.asarray(image, dtype=np.float32)

    return {
        "input_mean": float(np.mean(image)),
        "input_std": float(np.std(image)),
        "input_min": float(np.min(image)),
        "input_max": float(np.max(image)),
        "input_median": float(np.median(image)),
        "input_p05": float(np.percentile(image, 5)),
        "input_p95": float(np.percentile(image, 95)),
        "dark_pixel_percent": float(
            np.mean(image <= 0.05) * 100
        ),
        "bright_pixel_percent": float(
            np.mean(image >= 0.95) * 100
        ),
    }


# =========================================================
# U-NET PREDICTION STATISTICS
# =========================================================

def calculate_prediction_statistics(probability_map):

    result = {}

    for threshold in COMPARISON_THRESHOLDS:

        binary_mask = probability_map >= threshold

        result[
            f"mask_area_at_{threshold:.2f}_percent"
        ] = float(
            np.mean(binary_mask) * 100
        )

    result.update({
        "probability_mean": float(
            np.mean(probability_map)
        ),
        "probability_std": float(
            np.std(probability_map)
        ),
        "probability_min": float(
            np.min(probability_map)
        ),
        "probability_max": float(
            np.max(probability_map)
        ),
        "probability_median": float(
            np.median(probability_map)
        ),
        "probability_p95": float(
            np.percentile(probability_map, 95)
        ),
    })

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("FAILED VS SUCCESSFUL U-NET PREDICTION COMPARISON")
    print("=" * 70)

    for path in [
        VALIDATION_CSV,
        INVALID_CSV,
        UNET_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    validation_df = pd.read_csv(VALIDATION_CSV)
    invalid_df = pd.read_csv(INVALID_CSV)

    # Normalize image paths to absolute paths.
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

    # The four previously identified failed images.
    failed_df = validation_df[
        validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    if len(failed_df) != len(invalid_paths):
        print(
            "\nWarning: Number of matched failed images differs "
            "from invalid inspection records."
        )

    if failed_df.empty:
        raise RuntimeError(
            "Could not match failed images to validation records."
        )

    failed_df["comparison_group"] = "failed"

    # Successful control candidates from both datasets.
    control_candidates = validation_df[
        ~validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    # Select deterministic random samples from each source.
    controls = []

    for source in SOURCE_GROUPS:

        source_df = control_candidates[
            control_candidates["source_dataset"].astype(str)
            == source
        ]

        if source_df.empty:
            print(
                f"\nWarning: No control candidates for {source}"
            )
            continue

        sample_count = min(
            CONTROL_SAMPLE_SIZE,
            len(source_df)
        )

        selected = source_df.sample(
            n=sample_count,
            random_state=RANDOM_SEED
        ).copy()

        selected["comparison_group"] = (
            f"successful_{source}"
        )

        controls.append(selected)

        print(
            f"\nSelected {sample_count} controls from {source}"
        )

    if not controls:
        raise RuntimeError(
            "No successful control images were selected."
        )

    selected_df = pd.concat(
        [failed_df] + controls,
        ignore_index=True
    )

    print("\nFailed images:", len(failed_df))
    print(
        "Successful control images:",
        len(selected_df) - len(failed_df)
    )

    print("\nLoading balanced U-Net...")

    unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    results = []

    total_batches = (
        len(selected_df) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    # =====================================================
    # PROCESS SELECTED IMAGES
    # =====================================================

    for batch_number, start in enumerate(
        range(0, len(selected_df), BATCH_SIZE),
        start=1
    ):

        batch_df = selected_df.iloc[
            start:start + BATCH_SIZE
        ]

        images = []
        rows = []

        for _, row in batch_df.iterrows():

            image_path = Path(row["absolute_path"])

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            image = load_grayscale_image(image_path)

            images.append(image)
            rows.append(row)

        image_batch = np.stack(
            images
        ).astype(np.float32)

        probability_batch = unet(
            image_batch,
            training=False
        ).numpy()[..., 0]

        for index, row in enumerate(rows):

            input_image = image_batch[index, ..., 0]
            probability_map = probability_batch[index]

            input_stats = calculate_input_statistics(
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
                "processed_path": str(row["absolute_path"]),
                "final_mask_status": (
                    "empty"
                    if prediction_stats[
                        "mask_area_at_0.30_percent"
                    ] == 0
                    else "non_empty"
                ),
                **input_stats,
                **prediction_stats,
            }

            results.append(result)

        print(
            f"Processed batch {batch_number}/{total_batches}"
        )

    # =====================================================
    # SAVE PER-IMAGE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "failed_successful_per_image.csv",
        index=False
    )

    # =====================================================
    # SUMMARY BY COMPARISON GROUP
    # =====================================================

    metric_columns = [
        "input_mean",
        "input_std",
        "input_min",
        "input_max",
        "input_median",
        "input_p05",
        "input_p95",
        "dark_pixel_percent",
        "bright_pixel_percent",
        "probability_mean",
        "probability_std",
        "probability_max",
        "probability_p95",
        "mask_area_at_0.10_percent",
        "mask_area_at_0.20_percent",
        "mask_area_at_0.30_percent",
    ]

    summary = (
        results_df.groupby(
            "comparison_group",
            dropna=False
        )
        .agg(
            image_count=("processed_path", "size"),
            **{
                f"mean_{column}": (column, "mean")
                for column in metric_columns
            },
            empty_mask_count=(
                "final_mask_status",
                lambda values: (
                    values == "empty"
                ).sum()
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "comparison_group_summary.csv",
        index=False
    )

    # =====================================================
    # SAVE FAILED IMAGE DETAILS
    # =====================================================

    failed_results = results_df[
        results_df["comparison_group"] == "failed"
    ].copy()

    failed_results.to_csv(
        OUTPUT_DIR / "failed_image_details.csv",
        index=False
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)

    print("\nComparison group summary:")

    display_columns = [
        "comparison_group",
        "image_count",
        "mean_input_mean",
        "mean_input_std",
        "mean_probability_mean",
        "mean_probability_max",
        "mean_mask_area_at_0.10_percent",
        "mean_mask_area_at_0.20_percent",
        "mean_mask_area_at_0.30_percent",
        "empty_mask_count",
    ]

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    print("\nFailed image details:")

    print(
        failed_results[
            [
                "actual_class",
                "source_dataset",
                "input_mean",
                "input_std",
                "probability_mean",
                "probability_max",
                "mask_area_at_0.10_percent",
                "mask_area_at_0.20_percent",
                "mask_area_at_0.30_percent",
            ]
        ].to_string(index=False)
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- failed_successful_per_image.csv")
    print("- comparison_group_summary.csv")
    print("- failed_image_details.csv")


if __name__ == "__main__":
    main()
