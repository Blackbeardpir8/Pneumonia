
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data/processed/segmentation/manifest.csv"

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
    ROOT / "results/segmentation/spatial_intensity_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
TRAIN_SIZE = 40
COVID_SIZE = 20
BV_SIZE = 20
SEED = 42

GROUPS = [
    "unet_training",
    "failed",
    "successful_covid19_radiography",
    "successful_bacterial_viral_dataset",
]


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def load_image(path):

    image = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False,
    )

    original = image.numpy().squeeze()

    resized = tf.image.resize(
        tf.cast(image, tf.float32),
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
        antialias=False,
    )

    resized = resized.numpy().squeeze() / 255.0

    return original, resized


# =========================================================
# IMAGE SELECTION
# =========================================================

def select_images():

    manifest = pd.read_csv(MANIFEST)
    validation = pd.read_csv(VALIDATION_CSV)
    invalid = pd.read_csv(INVALID_CSV)

    train = manifest[
        manifest["split"].astype(str).str.lower() == "train"
    ].copy()

    train = train.sample(
        n=min(TRAIN_SIZE, len(train)),
        random_state=SEED,
    )

    train["absolute_path"] = train["processed_image"].apply(
        lambda p: ROOT / Path(p)
    )
    train["comparison_group"] = "unet_training"

    validation["absolute_path"] = validation[
        "processed_path"
    ].apply(
        lambda p: (
            Path(p)
            if Path(p).is_absolute()
            else ROOT / Path(p)
        )
    )

    invalid_paths = set(
        invalid["processed_path"].apply(
            lambda p: str(
                Path(p)
                if Path(p).is_absolute()
                else ROOT / Path(p)
            )
        )
    )

    failed = validation[
        validation["absolute_path"].apply(
            lambda p: str(p) in invalid_paths
        )
    ].copy()

    if len(failed) != len(invalid_paths):
        raise RuntimeError("Failed-image matching error.")

    failed["comparison_group"] = "failed"

    candidates = validation[
        ~validation["absolute_path"].apply(
            lambda p: str(p) in invalid_paths
        )
    ].copy()

    controls = []

    for source, count in [
        ("covid19_radiography", COVID_SIZE),
        ("bacterial_viral_dataset", BV_SIZE),
    ]:

        subset = candidates[
            candidates["source_dataset"].astype(str) == source
        ]

        selected = subset.sample(
            n=min(count, len(subset)),
            random_state=SEED,
        ).copy()

        selected["comparison_group"] = f"successful_{source}"

        controls.append(selected)

    training = train[[
        "absolute_path", "comparison_group"
    ]].copy()

    classification = pd.concat(
        [failed] + controls,
        ignore_index=True,
    )

    classification = classification[[
        "absolute_path",
        "comparison_group",
        "actual_class",
        "source_dataset",
    ]]

    return pd.concat(
        [training, classification],
        ignore_index=True,
        sort=False,
    )


# =========================================================
# SPATIAL ANALYSIS
# =========================================================

def calculate_spatial_metrics(image):

    height, width = image.shape

    # Divide image into a 3x3 grid.
    regions = {
        "top_left": image[:height//3, :width//3],
        "top_center": image[:height//3, width//3:2*width//3],
        "top_right": image[:height//3, 2*width//3:],

        "middle_left": image[height//3:2*height//3, :width//3],
        "middle_center": image[height//3:2*height//3, width//3:2*width//3],
        "middle_right": image[height//3:2*height//3, 2*width//3:],

        "bottom_left": image[2*height//3:, :width//3],
        "bottom_center": image[2*height//3:, width//3:2*width//3],
        "bottom_right": image[2*height//3:, 2*width//3:],
    }

    result = {}

    for name, region in regions.items():
        result[f"{name}_mean"] = float(np.mean(region))

    # Border intensity versus central intensity.
    border = np.concatenate([
        image[:10, :].ravel(),
        image[-10:, :].ravel(),
        image[:, :10].ravel(),
        image[:, -10:].ravel(),
    ])

    center = image[56:168, 56:168]

    result["border_mean"] = float(np.mean(border))
    result["center_mean"] = float(np.mean(center))
    result["border_center_difference"] = float(
        np.mean(border) - np.mean(center)
    )

    # Row and column profiles.
    row_profile = np.mean(image, axis=1)
    column_profile = np.mean(image, axis=0)

    result["top_row_mean"] = float(np.mean(row_profile[:56]))
    result["middle_row_mean"] = float(np.mean(row_profile[84:140]))
    result["bottom_row_mean"] = float(np.mean(row_profile[-56:]))

    result["left_column_mean"] = float(np.mean(column_profile[:56]))
    result["middle_column_mean"] = float(np.mean(column_profile[84:140]))
    result["right_column_mean"] = float(np.mean(column_profile[-56:]))

    return result


# =========================================================
# CREATE VISUALIZATION
# =========================================================

def create_visualization(records):

    fig, axes = plt.subplots(
        len(GROUPS),
        4,
        figsize=(16, 15),
    )

    for index, group in enumerate(GROUPS):

        group_records = [
            r for r in records
            if r["comparison_group"] == group
        ]

        if not group_records:
            continue

        images = np.stack([
            r["resized"] for r in group_records
        ])

        mean_image = np.mean(images, axis=0)

        row_profile = np.mean(mean_image, axis=1)
        column_profile = np.mean(mean_image, axis=0)

        axes[index, 0].imshow(
            group_records[0]["resized"],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[index, 0].set_title(
            f"{group}\nRepresentative image"
        )

        axes[index, 1].imshow(
            mean_image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[index, 1].set_title(
            "Mean spatial intensity"
        )

        axes[index, 2].plot(
            row_profile,
            label="Row intensity",
        )
        axes[index, 2].set_title(
            "Top-to-bottom intensity"
        )
        axes[index, 2].set_xlabel("Image row")
        axes[index, 2].set_ylim(0, 1)
        axes[index, 2].grid(alpha=0.2)

        axes[index, 3].plot(
            column_profile,
            label="Column intensity",
        )
        axes[index, 3].set_title(
            "Left-to-right intensity"
        )
        axes[index, 3].set_xlabel("Image column")
        axes[index, 3].set_ylim(0, 1)
        axes[index, 3].grid(alpha=0.2)

        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "spatial_intensity_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SPATIAL INTENSITY ANALYSIS")
    print("=" * 70)

    for path in [MANIFEST, VALIDATION_CSV, INVALID_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    selected = select_images()

    print("\nSelected groups:")
    print(selected["comparison_group"].value_counts())

    records = []
    statistics = []

    for index, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        path = Path(row["absolute_path"])

        if not path.exists():
            raise FileNotFoundError(path)

        original, resized = load_image(path)

        spatial = calculate_spatial_metrics(resized)

        record = {
            "comparison_group": row["comparison_group"],
            "image_name": path.name,
            "processed_path": str(path),
            "original_width": original.shape[1],
            "original_height": original.shape[0],
            **spatial,
        }

        records.append({
            **record,
            "resized": resized,
        })

        statistics.append(record)

        print(f"Processed {index}/{len(selected)}")

    # Individual statistics.
    stats_df = pd.DataFrame(statistics)

    stats_df.to_csv(
        OUTPUT_DIR / "spatial_intensity_per_image.csv",
        index=False,
    )

    # Group-level statistics.
    metric_columns = [
        column for column in stats_df.columns
        if column.endswith("_mean")
        or column in [
            "border_center_difference",
            "top_row_mean",
            "middle_row_mean",
            "bottom_row_mean",
            "left_column_mean",
            "middle_column_mean",
            "right_column_mean",
        ]
    ]

    summary = (
        stats_df.groupby("comparison_group")
        .agg(
            image_count=("image_name", "size"),
            **{
                f"mean_{column}": (column, "mean")
                for column in metric_columns
            },
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "spatial_intensity_group_summary.csv",
        index=False,
    )

    # Visualization.
    create_visualization(records)

    print("\n" + "=" * 70)
    print("SPATIAL ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nGroup summary:")
    print(summary.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- spatial_intensity_per_image.csv")
    print("- spatial_intensity_group_summary.csv")
    print("- spatial_intensity_comparison.png")


if __name__ == "__main__":
    main()
