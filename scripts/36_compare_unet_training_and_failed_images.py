
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

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data/processed/segmentation/manifest.csv"
)

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/segmentation"
    / "training_vs_failed_image_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
TRAIN_SAMPLE_SIZE = 40
COVID_CONTROL_SIZE = 20
BV_CONTROL_SIZE = 20
RANDOM_SEED = 42


# =========================================================
# IMAGE LOADING
# =========================================================

def load_original_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    return image.numpy().squeeze()


def prepare_image(image):

    image = tf.convert_to_tensor(
        image[..., np.newaxis],
        dtype=tf.float32,
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
        antialias=False,
    )

    image = image / 255.0

    return image.numpy().squeeze()


# =========================================================
# IMAGE STATISTICS
# =========================================================

def calculate_statistics(original, resized):

    resized = np.asarray(resized, dtype=np.float32)

    return {
        "original_height": original.shape[0],
        "original_width": original.shape[1],
        "resized_mean": float(np.mean(resized)),
        "resized_std": float(np.std(resized)),
        "resized_min": float(np.min(resized)),
        "resized_max": float(np.max(resized)),
        "resized_median": float(np.median(resized)),
        "resized_p01": float(np.percentile(resized, 1)),
        "resized_p05": float(np.percentile(resized, 5)),
        "resized_p25": float(np.percentile(resized, 25)),
        "resized_p75": float(np.percentile(resized, 75)),
        "resized_p95": float(np.percentile(resized, 95)),
        "resized_p99": float(np.percentile(resized, 99)),
        "dark_pixel_percent": float(
            np.mean(resized <= 0.05) * 100
        ),
        "bright_pixel_percent": float(
            np.mean(resized >= 0.95) * 100
        ),
    }


# =========================================================
# SELECT IMAGES
# =========================================================

def select_images():

    manifest = pd.read_csv(MANIFEST_PATH)

    validation_df = pd.read_csv(VALIDATION_CSV)
    invalid_df = pd.read_csv(INVALID_CSV)

    # -----------------------------------------------------
    # U-Net training images
    # -----------------------------------------------------

    train_df = manifest[
        manifest["split"].astype(str).str.lower() == "train"
    ].copy()

    if train_df.empty:
        raise RuntimeError("No U-Net training images found.")

    train_selected = train_df.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(train_df)),
        random_state=RANDOM_SEED,
    ).copy()

    train_selected["absolute_path"] = (
        train_selected["processed_image"].apply(
            lambda value: PROJECT_ROOT / Path(value)
        )
    )

    train_selected["comparison_group"] = "unet_training"

    # -----------------------------------------------------
    # Classification image paths
    # -----------------------------------------------------

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
            "Failed image matching count is incorrect."
        )

    failed_df["comparison_group"] = "failed"

    controls_df = validation_df[
        ~validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    # -----------------------------------------------------
    # Successful controls from both sources
    # -----------------------------------------------------

    controls = []

    source_settings = [
        ("covid19_radiography", COVID_CONTROL_SIZE),
        ("bacterial_viral_dataset", BV_CONTROL_SIZE),
    ]

    for source, sample_size in source_settings:

        source_df = controls_df[
            controls_df["source_dataset"].astype(str) == source
        ]

        if source_df.empty:
            raise RuntimeError(
                f"No successful controls found for {source}"
            )

        selected = source_df.sample(
            n=min(sample_size, len(source_df)),
            random_state=RANDOM_SEED,
        ).copy()

        selected["comparison_group"] = (
            f"successful_{source}"
        )

        controls.append(selected)

    classification_selected = pd.concat(
        [failed_df] + controls,
        ignore_index=True,
    )

    selected_frames = [
        train_selected[
            [
                "absolute_path",
                "comparison_group",
            ]
        ],
        classification_selected[
            [
                "absolute_path",
                "comparison_group",
                "source_dataset",
                "actual_class",
                "predicted_class",
            ]
        ],
    ]

    selected_df = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )

    return selected_df


# =========================================================
# CREATE VISUAL COMPARISON
# =========================================================

def create_visualization(records):

    group_order = [
        "unet_training",
        "failed",
        "successful_covid19_radiography",
        "successful_bacterial_viral_dataset",
    ]

    fig, axes = plt.subplots(
        len(group_order),
        3,
        figsize=(15, 16),
    )

    for row_index, group in enumerate(group_order):

        group_records = [
            record
            for record in records
            if record["comparison_group"] == group
        ]

        if not group_records:
            continue

        # Show first representative image.
        representative = group_records[0]

        axes[row_index, 0].imshow(
            representative["original"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[row_index, 0].set_title(
            f"{group}\n"
            f"Original: "
            f"{representative['original'].shape[1]}x"
            f"{representative['original'].shape[0]}"
        )

        axes[row_index, 1].imshow(
            representative["resized"],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 1].set_title(
            "Resized 224x224"
        )

        # Aggregate intensity histogram.
        all_pixels = np.concatenate([
            record["resized"].ravel()
            for record in group_records
        ])

        axes[row_index, 2].hist(
            all_pixels,
            bins=64,
            range=(0, 1),
        )

        axes[row_index, 2].set_title(
            f"Group intensity distribution\n"
            f"Images: {len(group_records)}"
        )

        axes[row_index, 2].set_xlabel(
            "Normalized intensity"
        )

        axes[row_index, 2].set_ylabel(
            "Pixel count"
        )

        axes[row_index, 2].grid(
            True,
            alpha=0.2,
        )

        for axis in axes[row_index]:
            if axis != axes[row_index, 2]:
                axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_failed_distribution_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("U-NET TRAINING VS FAILED IMAGE ANALYSIS")
    print("=" * 70)

    for path in [
        MANIFEST_PATH,
        VALIDATION_CSV,
        INVALID_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    selected_df = select_images()

    print("\nSelected image counts:")
    print(
        selected_df["comparison_group"]
        .value_counts()
        .to_string()
    )

    records = []

    # -----------------------------------------------------
    # Process all selected images
    # -----------------------------------------------------

    for index, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1,
    ):

        image_path = Path(row["absolute_path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        original = load_original_image(image_path)

        resized = prepare_image(original)

        stats = calculate_statistics(
            original,
            resized,
        )

        record = {
            "comparison_group": row["comparison_group"],
            "source_dataset": row.get("source_dataset"),
            "actual_class": row.get("actual_class"),
            "predicted_class": row.get("predicted_class"),
            "image_name": image_path.name,
            "processed_path": str(image_path),
            **stats,
        }

        records.append({
            **record,
            "original": original,
            "resized": resized,
        })

        print(
            f"Processed {index}/{len(selected_df)}"
        )

    # -----------------------------------------------------
    # Save statistics
    # -----------------------------------------------------

    stats_df = pd.DataFrame([
        {
            key: value
            for key, value in record.items()
            if key not in ["original", "resized"]
        }
        for record in records
    ])

    stats_df.to_csv(
        OUTPUT_DIR / "image_distribution_statistics.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Group summary
    # -----------------------------------------------------

    summary = (
        stats_df.groupby("comparison_group")
        .agg(
            image_count=("image_name", "size"),
            mean_intensity=("resized_mean", "mean"),
            mean_contrast=("resized_std", "mean"),
            mean_median=("resized_median", "mean"),
            mean_p05=("resized_p05", "mean"),
            mean_p95=("resized_p95", "mean"),
            mean_dark_pixels=("dark_pixel_percent", "mean"),
            mean_bright_pixels=("bright_pixel_percent", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "group_distribution_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    create_visualization(records)

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nGroup summary:")
    print(summary.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- image_distribution_statistics.csv")
    print("- group_distribution_summary.csv")
    print("- training_failed_distribution_comparison.png")


if __name__ == "__main__":
    main()
