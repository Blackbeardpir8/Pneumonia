
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = (
    ROOT / "data/processed/segmentation/manifest.csv"
)

VALIDATION_CSV = (
    ROOT / "results/classification/validation_evaluation"
    / "validation_predictions.csv"
)

INVALID_CSV = (
    ROOT / "results/classification/validation_gradcam_analysis"
    / "invalid_mask_inspection"
    / "invalid_mask_inspection_summary.csv"
)

OUTPUT_DIR = (
    ROOT / "results/segmentation"
    / "lung_region_intensity_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
TRAIN_SAMPLE_SIZE = 40
COVID_CONTROL_SIZE = 20
RANDOM_SEED = 42


# =========================================================
# IMAGE LOADING
# =========================================================

def load_grayscale(path):

    image_bytes = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False,
    )

    return image.numpy().squeeze()


def resize_for_analysis(image):

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

    return (image.numpy().squeeze() / 255.0).astype(
        np.float32
    )


# =========================================================
# STATISTICS
# =========================================================

def calculate_intensity_statistics(values):

    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        raise ValueError("Cannot calculate statistics on empty region.")

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "dark_pixel_percent": float(
            np.mean(values <= 0.05) * 100
        ),
        "bright_pixel_percent": float(
            np.mean(values >= 0.95) * 100
        ),
    }


def calculate_global_statistics(image):

    stats = calculate_intensity_statistics(
        image.ravel()
    )

    return {
        f"global_{key}": value
        for key, value in stats.items()
    }


# =========================================================
# SELECT IMAGES
# =========================================================

def select_images():

    manifest = pd.read_csv(MANIFEST)
    validation = pd.read_csv(VALIDATION_CSV)
    invalid = pd.read_csv(INVALID_CSV)

    required_manifest_columns = [
        "pair_id",
        "split",
        "processed_image",
        "processed_mask",
    ]

    for column in required_manifest_columns:
        if column not in manifest.columns:
            raise ValueError(
                f"Missing manifest column: {column}"
            )

    # -----------------------------------------------------
    # U-Net training images with ground-truth masks
    # -----------------------------------------------------

    train = manifest[
        manifest["split"].astype(str).str.lower() == "train"
    ].copy()

    train = train.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(train)),
        random_state=RANDOM_SEED,
    )

    train["absolute_path"] = train["processed_image"].apply(
        lambda value: ROOT / Path(value)
    )

    train["absolute_mask_path"] = train[
        "processed_mask"
    ].apply(
        lambda value: ROOT / Path(value)
    )

    train["comparison_group"] = "unet_training"

    # -----------------------------------------------------
    # Classification images
    # -----------------------------------------------------

    validation["absolute_path"] = validation[
        "processed_path"
    ].apply(
        lambda value: (
            Path(value)
            if Path(value).is_absolute()
            else ROOT / Path(value)
        )
    )

    invalid_paths = set(
        invalid["processed_path"].apply(
            lambda value: str(
                Path(value)
                if Path(value).is_absolute()
                else ROOT / Path(value)
            )
        )
    )

    failed = validation[
        validation["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    if len(failed) != len(invalid_paths):
        raise RuntimeError(
            "Failed image matching count is incorrect."
        )

    failed["comparison_group"] = "failed"

    successful = validation[
        ~validation["absolute_path"].apply(
            lambda path: str(path) in invalid_paths
        )
    ].copy()

    # -----------------------------------------------------
    # Successful COVID controls
    # -----------------------------------------------------

    covid = successful[
        successful["source_dataset"].astype(str)
        == "covid19_radiography"
    ]

    if covid.empty:
        raise RuntimeError(
            "No successful COVID controls found."
        )

    covid = covid.sample(
        n=min(COVID_CONTROL_SIZE, len(covid)),
        random_state=RANDOM_SEED,
    ).copy()

    covid["comparison_group"] = "successful_covid"

    return train, failed, covid


# =========================================================
# TRAINING IMAGE: TRUE LUNG VS BACKGROUND
# =========================================================

def analyze_training_image(image_path, mask_path):

    image = load_grayscale(image_path)

    mask = load_grayscale(mask_path)

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected training image shape: {image.shape}"
        )

    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected mask shape: {mask.shape}"
        )

    image = image.astype(np.float32) / 255.0

    true_lung_mask = mask >= 128

    if not np.any(true_lung_mask):
        raise ValueError(
            f"Empty ground-truth mask: {mask_path}"
        )

    background_mask = ~true_lung_mask

    lung_stats = calculate_intensity_statistics(
        image[true_lung_mask]
    )

    background_stats = calculate_intensity_statistics(
        image[background_mask]
    )

    global_stats = calculate_global_statistics(image)

    result = {
        **{
            f"lung_{key}": value
            for key, value in lung_stats.items()
        },
        **{
            f"background_{key}": value
            for key, value in background_stats.items()
        },
        **global_stats,
        "ground_truth_lung_area_percent": float(
            np.mean(true_lung_mask) * 100
        ),
    }

    return image, true_lung_mask, result


# =========================================================
# CLASSIFICATION IMAGE: WHOLE IMAGE ONLY
# =========================================================

def analyze_classification_image(image_path):

    original = load_grayscale(image_path)

    resized = resize_for_analysis(original)

    result = calculate_global_statistics(resized)

    return original, resized, result


# =========================================================
# VISUALIZATION
# =========================================================

def create_training_mask_visualization(training_examples):

    count = min(6, len(training_examples))

    fig, axes = plt.subplots(
        count,
        3,
        figsize=(12, count * 3),
        squeeze=False,
    )

    for index, example in enumerate(
        training_examples[:count]
    ):

        image = example["image"]
        mask = example["mask"]

        axes[index, 0].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[index, 0].set_title("Training X-ray")

        axes[index, 1].imshow(
            mask,
            cmap="gray",
        )
        axes[index, 1].set_title("Ground-truth lung mask")

        axes[index, 2].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[index, 2].contour(
            mask.astype(float),
            levels=[0.5],
        )

        axes[index, 2].set_title(
            "Ground-truth mask overlay"
        )

        for axis in axes[index]:
            axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_ground_truth_mask_examples.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_intensity_histograms(training_values,
                                failed_values,
                                covid_values):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    # Ground-truth training regions.
    axes[0].hist(
        training_values["lung"],
        bins=60,
        range=(0, 1),
        alpha=0.6,
        label="Training lung pixels",
    )

    axes[0].hist(
        training_values["background"],
        bins=60,
        range=(0, 1),
        alpha=0.6,
        label="Training background pixels",
    )

    axes[0].set_title(
        "U-Net training: true lung vs background"
    )

    axes[0].set_xlabel("Normalized intensity")
    axes[0].set_ylabel("Pixel count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Whole-image comparison.
    axes[1].hist(
        failed_values,
        bins=60,
        range=(0, 1),
        alpha=0.6,
        label="Failed images",
    )

    axes[1].hist(
        covid_values,
        bins=60,
        range=(0, 1),
        alpha=0.6,
        label="Successful COVID images",
    )

    axes[1].set_title(
        "Classification images: whole-image intensities"
    )

    axes[1].set_xlabel("Normalized intensity")
    axes[1].set_ylabel("Pixel count")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "lung_background_intensity_histograms.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("LUNG REGION VS BACKGROUND INTENSITY ANALYSIS")
    print("=" * 70)

    for path in [
        MANIFEST,
        VALIDATION_CSV,
        INVALID_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    train, failed, covid = select_images()

    print("\nSelected images:")
    print("U-Net training:", len(train))
    print("Failed:", len(failed))
    print("Successful COVID:", len(covid))

    training_results = []
    training_examples = []

    training_lung_pixels = []
    training_background_pixels = []

    # -----------------------------------------------------
    # Analyze training images using actual masks
    # -----------------------------------------------------

    for index, (_, row) in enumerate(
        train.iterrows(),
        start=1,
    ):

        image_path = Path(row["absolute_path"])
        mask_path = Path(row["absolute_mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image, mask, metrics = analyze_training_image(
            image_path,
            mask_path,
        )

        training_lung_pixels.append(
            image[mask].ravel()
        )

        training_background_pixels.append(
            image[~mask].ravel()
        )

        training_results.append({
            "comparison_group": "unet_training",
            "image_name": image_path.name,
            "processed_path": str(image_path),
            **metrics,
        })

        if index <= 6:
            training_examples.append({
                "image": image,
                "mask": mask,
            })

        print(
            f"Training image {index}/{len(train)}"
        )

    # -----------------------------------------------------
    # Analyze failed images
    # -----------------------------------------------------

    failed_results = []
    failed_pixels = []

    for index, (_, row) in enumerate(
        failed.iterrows(),
        start=1,
    ):

        path = Path(row["absolute_path"])

        original, resized, metrics = (
            analyze_classification_image(path)
        )

        failed_pixels.append(resized.ravel())

        failed_results.append({
            "comparison_group": "failed",
            "image_name": path.name,
            "processed_path": str(path),
            "actual_class": row.get("actual_class"),
            **metrics,
        })

        print(
            f"Failed image {index}/{len(failed)}"
        )

    # -----------------------------------------------------
    # Analyze successful COVID images
    # -----------------------------------------------------

    covid_results = []
    covid_pixels = []

    for index, (_, row) in enumerate(
        covid.iterrows(),
        start=1,
    ):

        path = Path(row["absolute_path"])

        original, resized, metrics = (
            analyze_classification_image(path)
        )

        covid_pixels.append(resized.ravel())

        covid_results.append({
            "comparison_group": "successful_covid",
            "image_name": path.name,
            "processed_path": str(path),
            "actual_class": row.get("actual_class"),
            **metrics,
        })

        print(
            f"Successful COVID image {index}/{len(covid)}"
        )

    # -----------------------------------------------------
    # Save per-image statistics
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        training_results
        + failed_results
        + covid_results
    )

    results_df.to_csv(
        OUTPUT_DIR / "lung_region_per_image.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Aggregate true training lung/background pixels
    # -----------------------------------------------------

    all_lung_pixels = np.concatenate(
        training_lung_pixels
    )

    all_background_pixels = np.concatenate(
        training_background_pixels
    )

    region_summary = pd.DataFrame([
        {
            "region": "training_true_lung",
            **calculate_intensity_statistics(
                all_lung_pixels
            ),
            "pixel_count": len(all_lung_pixels),
        },
        {
            "region": "training_background",
            **calculate_intensity_statistics(
                all_background_pixels
            ),
            "pixel_count": len(all_background_pixels),
        },
        {
            "region": "failed_whole_image",
            **calculate_intensity_statistics(
                np.concatenate(failed_pixels)
            ),
            "pixel_count": sum(
                len(values) for values in failed_pixels
            ),
        },
        {
            "region": "successful_covid_whole_image",
            **calculate_intensity_statistics(
                np.concatenate(covid_pixels)
            ),
            "pixel_count": sum(
                len(values) for values in covid_pixels
            ),
        },
    ])

    region_summary.to_csv(
        OUTPUT_DIR / "lung_background_region_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Visualizations
    # -----------------------------------------------------

    create_training_mask_visualization(
        training_examples
    )

    create_intensity_histograms(
        {
            "lung": all_lung_pixels,
            "background": all_background_pixels,
        },
        failed_pixels,
        covid_pixels,
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LUNG REGION ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nRegion intensity summary:")

    print(
        region_summary[
            [
                "region",
                "pixel_count",
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

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- lung_region_per_image.csv")
    print("- lung_background_region_summary.csv")
    print("- training_ground_truth_mask_examples.png")
    print("- lung_background_intensity_histograms.png")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/38_analyze_lung_region_intensity.py