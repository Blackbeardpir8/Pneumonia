
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
    / "results/classification"
    / "failed_image_intensity_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

CONTROL_SAMPLE_SIZE = 5
RANDOM_SEED = 42

SOURCE_GROUPS = [
    "covid19_radiography",
    "bacterial_viral_dataset",
]


# =========================================================
# LOAD ORIGINAL GRAYSCALE IMAGE
# =========================================================

def load_grayscale_image(image_path):

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    return image.numpy().squeeze()


# =========================================================
# IMAGE STATISTICS
# =========================================================

def calculate_statistics(image):

    image = np.asarray(image, dtype=np.float32)

    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "median": float(np.median(image)),
        "p01": float(np.percentile(image, 1)),
        "p05": float(np.percentile(image, 5)),
        "p25": float(np.percentile(image, 25)),
        "p75": float(np.percentile(image, 75)),
        "p95": float(np.percentile(image, 95)),
        "p99": float(np.percentile(image, 99)),
        "dark_pixel_percent": float(
            np.mean(image <= 25.5) * 100
        ),
        "bright_pixel_percent": float(
            np.mean(image >= 229.5) * 100
        ),
    }


# =========================================================
# CONTRAST ENHANCEMENT FOR VISUAL INSPECTION ONLY
# =========================================================

def enhance_contrast_for_preview(image):

    image = image.astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)

    enhanced = (image - low) / (high - low)

    enhanced = np.clip(enhanced, 0, 1)

    return (enhanced * 255).astype(np.uint8)


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

    # Select the four failed images.
    failed_df = validation_df[
        validation_df["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    if len(failed_df) != len(invalid_paths):
        raise RuntimeError(
            "Failed image matching count is incorrect."
        )

    if failed_df.empty:
        raise RuntimeError("No failed images found.")

    failed_df["comparison_group"] = "failed"

    # Select successful control candidates.
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
                f"No control images found for {source}"
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
# CREATE VISUAL COMPARISON
# =========================================================

def create_visualization(selected_df):

    total_images = len(selected_df)

    fig, axes = plt.subplots(
        total_images,
        3,
        figsize=(15, total_images * 3.5),
        squeeze=False,
    )

    statistics = []

    for index, (_, row) in enumerate(
        selected_df.iterrows()
    ):

        image_path = Path(row["absolute_path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = load_grayscale_image(image_path)

        enhanced = enhance_contrast_for_preview(image)

        stats = calculate_statistics(image)

        stats.update({
            "comparison_group": row["comparison_group"],
            "source_dataset": row["source_dataset"],
            "actual_class": row.get("actual_class"),
            "predicted_class": row.get("predicted_class"),
            "processed_path": str(image_path),
        })

        statistics.append(stats)

        title = (
            f"{row['comparison_group']}\n"
            f"Actual: {row.get('actual_class')} | "
            f"Predicted: {row.get('predicted_class')}\n"
            f"{image_path.name}"
        )

        # Original image.
        axes[index, 0].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[index, 0].set_title(
            title,
            fontsize=8,
        )

        axes[index, 0].axis("off")

        # Contrast-enhanced preview.
        axes[index, 1].imshow(
            enhanced,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[index, 1].set_title(
            "Contrast-enhanced preview\n"
            f"Mean: {stats['mean']:.1f} | "
            f"Std: {stats['std']:.1f}",
            fontsize=9,
        )

        axes[index, 1].axis("off")

        # Original image histogram.
        axes[index, 2].hist(
            image.flatten(),
            bins=64,
            range=(0, 256),
        )

        axes[index, 2].set_title(
            "Original grayscale histogram",
            fontsize=9,
        )

        axes[index, 2].set_xlim(0, 255)
        axes[index, 2].set_xlabel("Pixel intensity")
        axes[index, 2].set_ylabel("Pixel count")
        axes[index, 2].grid(
            True,
            alpha=0.2,
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "image_intensity_comparison.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return pd.DataFrame(statistics)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("FAILED IMAGE INTENSITY AND CONTRAST ANALYSIS")
    print("=" * 70)

    for path in [VALIDATION_CSV, INVALID_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    selected_df = select_images()

    print("\nSelected image counts:")
    print(
        selected_df["comparison_group"]
        .value_counts()
        .to_string()
    )

    print("\nGenerating image comparison...")

    stats_df = create_visualization(selected_df)

    # Save selected image information and statistics.
    stats_df.to_csv(
        OUTPUT_DIR / "image_intensity_statistics.csv",
        index=False,
    )

    selected_df[
        [
            "comparison_group",
            "source_dataset",
            "actual_class",
            "predicted_class",
            "absolute_path",
        ]
    ].to_csv(
        OUTPUT_DIR / "selected_images.csv",
        index=False,
    )

    # Print failed image statistics.
    failed_stats = stats_df[
        stats_df["comparison_group"] == "failed"
    ]

    print("\nFailed image statistics:")

    print(
        failed_stats[
            [
                "actual_class",
                "width",
                "height",
                "mean",
                "std",
                "median",
                "p05",
                "p95",
                "dark_pixel_percent",
                "bright_pixel_percent",
            ]
        ].to_string(index=False)
    )

    # Print group averages.
    summary = (
        stats_df.groupby("comparison_group")
        .agg(
            image_count=("processed_path", "size"),
            mean_intensity=("mean", "mean"),
            mean_contrast=("std", "mean"),
            mean_median=("median", "mean"),
            mean_p05=("p05", "mean"),
            mean_p95=("p95", "mean"),
            mean_dark_pixels=("dark_pixel_percent", "mean"),
            mean_bright_pixels=("bright_pixel_percent", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "group_intensity_summary.csv",
        index=False,
    )

    print("\nGroup summary:")

    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- image_intensity_comparison.png")
    print("- image_intensity_statistics.csv")
    print("- group_intensity_summary.csv")
    print("- selected_images.csv")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/33_inspect_failed_image_intensity.py
