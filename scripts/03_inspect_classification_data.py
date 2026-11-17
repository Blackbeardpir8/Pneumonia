
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image, UnidentifiedImageError

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
    CLASS_TO_INDEX,
)

# ============================================================
# 1. PATHS
# ============================================================

MANIFEST_PATH = CLASSIFICATION_DATA_DIR / "manifest.csv"

INSPECTION_DIR = CLASSIFICATION_RESULTS_DIR / "inspection"

# ============================================================
# 2. LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "split",
        "class_name",
        "class_index",
        "processed_path",
        "sha256",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {missing_columns}"
        )

    print("\nManifest loaded successfully.")
    print("Total records:", len(df))

    return df


# ============================================================
# 3. CHECK IMAGE PATHS AND LABELS
# ============================================================

def inspect_image_records(df):

    missing_files = []
    invalid_images = []
    label_errors = []
    dimension_records = []
    mode_counts = Counter()

    for index, row in df.iterrows():

        image_path = Path(row["processed_path"])

        # Check image existence.
        if not image_path.is_file():
            missing_files.append(str(image_path))
            continue

        # Check class index.
        expected_index = CLASS_TO_INDEX.get(row["class_name"])

        if expected_index is None or int(row["class_index"]) != expected_index:
            label_errors.append({
                "path": str(image_path),
                "class_name": row["class_name"],
                "class_index": row["class_index"],
            })

        # Check folder and manifest class agreement.
        if image_path.parent.name != row["class_name"]:
            label_errors.append({
                "path": str(image_path),
                "error": "Folder and manifest class mismatch",
            })

        try:
            with Image.open(image_path) as image:

                image.verify()

            with Image.open(image_path) as image:

                image.load()

                width, height = image.size
                mode = image.mode

                dimension_records.append({
                    "path": str(image_path),
                    "split": row["split"],
                    "class_name": row["class_name"],
                    "width": width,
                    "height": height,
                    "mode": mode,
                })

                mode_counts[mode] += 1

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            invalid_images.append({
                "path": str(image_path),
                "error": str(error),
            })

        if (index + 1) % 2000 == 0:
            print(
                f"Inspected {index + 1}/{len(df)} records..."
            )

    return (
        missing_files,
        invalid_images,
        label_errors,
        dimension_records,
        mode_counts,
    )


# ============================================================
# 4. CHECK DUPLICATE HASHES ACROSS SPLITS
# ============================================================

def inspect_split_leakage(df):

    duplicate_hashes = df[
        df.duplicated(subset=["sha256"], keep=False)
    ]

    leakage_records = []

    for sha256, group in duplicate_hashes.groupby("sha256"):

        splits = sorted(group["split"].unique())

        if len(splits) > 1:

            leakage_records.append({
                "sha256": sha256,
                "splits": ", ".join(splits),
                "image_count": len(group),
            })

    print("\nSplit leakage inspection completed.")

    print(
        "Duplicate hashes across different splits:",
        len(leakage_records),
    )

    return leakage_records


# ============================================================
# 5. GENERATE REPORTS
# ============================================================

def generate_reports(
    df,
    dimension_records,
    missing_files,
    invalid_images,
    label_errors,
    leakage_records,
):

    INSPECTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Class distribution.
    distribution = pd.crosstab(
        df["split"],
        df["class_name"]
    ).reindex(
        index=["train", "val", "test"],
        columns=CLASS_NAMES,
        fill_value=0
    )

    distribution.to_csv(
        INSPECTION_DIR / "class_distribution.csv"
    )

    # Image dimensions and modes.
    dimensions_df = pd.DataFrame(dimension_records)

    dimensions_df.to_csv(
        INSPECTION_DIR / "image_dimensions.csv",
        index=False
    )

    # Problem reports.
    pd.DataFrame(
        {"missing_path": missing_files}
    ).to_csv(
        INSPECTION_DIR / "missing_images.csv",
        index=False
    )

    pd.DataFrame(invalid_images).to_csv(
        INSPECTION_DIR / "invalid_images.csv",
        index=False
    )

    pd.DataFrame(label_errors).to_csv(
        INSPECTION_DIR / "label_errors.csv",
        index=False
    )

    pd.DataFrame(leakage_records).to_csv(
        INSPECTION_DIR / "split_leakage.csv",
        index=False
    )

    print("\nReports generated in:")
    print(INSPECTION_DIR)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PNEUMONIA CDSS - CLASSIFICATION DATA INSPECTION")
    print("=" * 65)

    df = load_manifest()

    print("\nClass distribution:")
    print(
        df.groupby(["split", "class_name"]).size().unstack(
            fill_value=0
        )
    )

    print("\nInspecting image files...")

    (
        missing_files,
        invalid_images,
        label_errors,
        dimension_records,
        mode_counts,
    ) = inspect_image_records(df)

    leakage_records = inspect_split_leakage(df)

    print("\nIMAGE INSPECTION RESULTS")
    print("-" * 45)
    print("Total manifest records:", len(df))
    print("Missing images:", len(missing_files))
    print("Invalid images:", len(invalid_images))
    print("Label errors:", len(label_errors))
    print("Duplicate hashes across splits:", len(leakage_records))

    print("\nImage modes:")
    print(dict(mode_counts))

    if dimension_records:

        dimensions_df = pd.DataFrame(dimension_records)

        print("\nImage dimension distribution:")
        print(
            dimensions_df.groupby(
                ["width", "height"]
            ).size().sort_values(ascending=False).head(15)
        )

    generate_reports(
        df,
        dimension_records,
        missing_files,
        invalid_images,
        label_errors,
        leakage_records,
    )

    print("\n" + "=" * 65)

    if (
        missing_files
        or invalid_images
        or label_errors
        or leakage_records
    ):
        print("INSPECTION COMPLETED WITH WARNINGS.")
    else:
        print("DATASET INSPECTION PASSED.")

    print("=" * 65)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/03_inspect_classification_data.py