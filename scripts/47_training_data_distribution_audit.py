from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import CLASSIFICATION_DATA_DIR, CLASSIFICATION_RESULTS_DIR

MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"
OUTPUT_DIR = CLASSIFICATION_RESULTS_DIR / "training_data_distribution_audit"

# This audit intentionally reads TRAIN images only.
# Validation and test images are not opened.
SPLIT_TO_AUDIT = "train"

REQUIRED_COLUMNS = ["split", "class_name", "source_dataset", "processed_path"]


def resolve_image_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path

    # Manifest paths may be relative to the project root or classification folder.
    candidates = [ROOT / path, CLASSIFICATION_DATA_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_image_statistics(path):
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False,
    )
    image = tf.cast(image, tf.float32) / 255.0

    shape = tf.shape(image)
    height = int(shape[0].numpy())
    width = int(shape[1].numpy())

    gray = tf.image.rgb_to_grayscale(image)
    values = tf.reshape(gray, [-1])

    stats = {
        "height": height,
        "width": width,
        "mean_intensity": float(tf.reduce_mean(values).numpy()),
        "std_intensity": float(tf.math.reduce_std(values).numpy()),
        "median_intensity": float(tfp_percentile(values, 50)),
        "p05_intensity": float(tfp_percentile(values, 5)),
        "p95_intensity": float(tfp_percentile(values, 95)),
        "dark_pixel_fraction": float(
            tf.reduce_mean(tf.cast(values < 0.10, tf.float32)).numpy()
        ),
        "bright_pixel_fraction": float(
            tf.reduce_mean(tf.cast(values > 0.90, tf.float32)).numpy()
        ),
    }
    return stats


def tfp_percentile(values, percentile):
    # tfp is deliberately not required; NumPy computes the small per-image summary.
    return np.percentile(values.numpy(), percentile)


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}. "
            "Move or archive the existing results before rerunning."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {missing}\n"
            f"Available columns: {list(manifest.columns)}"
        )

    train_df = manifest[manifest["split"].astype(str).str.lower() == SPLIT_TO_AUDIT].copy()
    if train_df.empty:
        raise ValueError("No rows found for split='train' in the manifest.")

    # Class/source counts are calculated from the training manifest.
    count_table = (
        train_df.groupby(["source_dataset", "class_name"], dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values(["source_dataset", "class_name"])
    )
    count_table.to_csv(OUTPUT_DIR / "training_class_source_counts.csv", index=False)

    print("\nTRAINING IMAGE COUNTS BY SOURCE AND CLASS")
    print(count_table.to_string(index=False))

    # Inspect every training image's original dimensions and grayscale intensity.
    records = []
    failures = []

    total = len(train_df)
    for index, row in enumerate(train_df.itertuples(index=False), start=1):
        image_path = resolve_image_path(getattr(row, "processed_path"))
        record = {
            "source_dataset": getattr(row, "source_dataset"),
            "class_name": getattr(row, "class_name"),
            "image_path": str(image_path),
        }

        try:
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            record.update(read_image_statistics(image_path))
            records.append(record)
        except Exception as exc:
            failures.append({
                "source_dataset": getattr(row, "source_dataset"),
                "class_name": getattr(row, "class_name"),
                "image_path": str(image_path),
                "error": repr(exc),
            })

        if index % 500 == 0 or index == total:
            print(f"Processed {index:,}/{total:,} training images")

    image_df = pd.DataFrame(records)
    failures_df = pd.DataFrame(failures)

    if image_df.empty:
        raise RuntimeError("No training images could be inspected.")

    image_df.to_csv(OUTPUT_DIR / "training_image_level_statistics.csv", index=False)

    group_columns = ["source_dataset", "class_name"]
    metrics = [
        "height",
        "width",
        "mean_intensity",
        "std_intensity",
        "median_intensity",
        "p05_intensity",
        "p95_intensity",
        "dark_pixel_fraction",
        "bright_pixel_fraction",
    ]

    summary = (
        image_df.groupby(group_columns, dropna=False)[metrics]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT_DIR / "training_source_class_image_statistics.csv", index=False)

    # Resolution frequency makes it easy to see whether one source/class
    # has a different native image-size pattern.
    resolution_counts = (
        image_df.groupby(
            ["source_dataset", "class_name", "height", "width"],
            dropna=False,
        )
        .size()
        .reset_index(name="image_count")
        .sort_values(
            ["source_dataset", "class_name", "image_count"],
            ascending=[True, True, False],
        )
    )
    resolution_counts.to_csv(
        OUTPUT_DIR / "training_resolution_frequencies.csv",
        index=False,
    )

    failures_df.to_csv(OUTPUT_DIR / "training_image_read_failures.csv", index=False)

    print("\nSOURCE/CLASS IMAGE STATISTICS (MEANS)")
    mean_columns = group_columns + [
        "height_mean",
        "width_mean",
        "mean_intensity_mean",
        "std_intensity_mean",
        "median_intensity_mean",
        "p05_intensity_mean",
        "p95_intensity_mean",
        "dark_pixel_fraction_mean",
        "bright_pixel_fraction_mean",
    ]
    print(summary[mean_columns].to_string(index=False))

    print(f"\nTraining rows in manifest: {total:,}")
    print(f"Images successfully inspected: {len(image_df):,}")
    print(f"Image read failures: {len(failures_df):,}")
    print(f"\nSaved audit files to:\n{OUTPUT_DIR}")
    print("\nOnly split='train' was inspected. Validation and test images were not opened.")


if __name__ == "__main__":
    main()


