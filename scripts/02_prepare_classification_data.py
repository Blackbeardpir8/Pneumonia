"""
Prepare the four-class chest X-ray classification dataset.

Classes:
    Normal
    Viral
    Bacterial
    COVID

Responsibilities:
    - Load source images and labels.
    - Validate image files.
    - Detect exact duplicates using SHA256.
    - Prevent conflicting labels.
    - Create stratified train/validation/test splits.
    - Generate dataset manifests and audit reports.
"""


import hashlib
import json
import shutil
import sys

from collections import defaultdict
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from aiml.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    CLASSIFICATION_DATA_DIR,
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    BACTERIAL_VIRAL_CLASS_MAPPING,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    IMAGE_SIZE,
)


# ============================================================
# 1. SOURCE DATASET PATHS
# ============================================================

# Bacterial, Viral and Normal dataset
BACTERIAL_VIRAL_CSV = (
    RAW_DATA_DIR / "archive (3)" / "labels_train.csv"
)

BACTERIAL_VIRAL_IMAGE_DIR = (
    RAW_DATA_DIR / "archive (3)" /
    "train_images" / "train_images"
)

# COVID-19 Radiography dataset
COVID_IMAGE_DIR = (
    RAW_DATA_DIR / "archive" /
    "COVID-19_Radiography_Dataset" /
    "COVID" / "images"
)

NORMAL_IMAGE_DIR = (
    RAW_DATA_DIR / "archive" /
    "COVID-19_Radiography_Dataset" /
    "Normal" / "images"
)

VIRAL_IMAGE_DIR = (
    RAW_DATA_DIR / "archive" /
    "COVID-19_Radiography_Dataset" /
    "Viral Pneumonia" / "images"
)


# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

AUDIT_DIR = CLASSIFICATION_RESULTS_DIR / "dataset_audit"

TRAIN_DIR = CLASSIFICATION_DATA_DIR / "train"
VAL_DIR = CLASSIFICATION_DATA_DIR / "val"
TEST_DIR = CLASSIFICATION_DATA_DIR / "test"


# ============================================================
# 3. SHA256 HASH FUNCTION
# ============================================================

def calculate_sha256(image_path):
    """
    Calculate SHA256 hash of an image file.

    This helps identify exact duplicate images,
    even when their filenames are different.
    """

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 4. IMAGE VALIDATION FUNCTION
# ============================================================

def is_valid_image(image_path):
    """
    Check whether an image can be opened and decoded.
    """

    try:
        with Image.open(image_path) as image:
            image.verify()

        # Reopen to verify that the image can be decoded.
        with Image.open(image_path) as image:
            image.load()

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return False

# ============================================================
# 5. COLLECT IMAGE RECORDS
# ============================================================

def collect_classification_records():
    """
    Collect image paths and verified class labels
    from the selected raw datasets.

    Returns:
        list: Image records containing path, label,
              class index, and source dataset.
    """

    records = []

    # --------------------------------------------------------
    # A. BACTERIAL, VIRAL AND NORMAL DATASET
    # --------------------------------------------------------

    print("\nLoading Bacterial/Viral dataset...")

    if not BACTERIAL_VIRAL_CSV.exists():
        raise FileNotFoundError(
            f"Label CSV not found: {BACTERIAL_VIRAL_CSV}"
        )

    if not BACTERIAL_VIRAL_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {BACTERIAL_VIRAL_IMAGE_DIR}"
        )

    labels_df = pd.read_csv(BACTERIAL_VIRAL_CSV)

    required_columns = {"file_name", "class_id"}

    if not required_columns.issubset(labels_df.columns):
        raise ValueError(
            "CSV must contain file_name and class_id columns."
        )

    if labels_df["file_name"].duplicated().any():
        raise ValueError(
            "Duplicate filenames found in labels_train.csv."
        )

    for _, row in labels_df.iterrows():

        file_name = str(row["file_name"])
        class_id = str(row["class_id"]).strip()

        if class_id not in BACTERIAL_VIRAL_CLASS_MAPPING:
            raise ValueError(
                f"Unknown class ID {class_id} for {file_name}"
            )

        class_name = BACTERIAL_VIRAL_CLASS_MAPPING[class_id]

        image_path = BACTERIAL_VIRAL_IMAGE_DIR / file_name

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image listed in CSV is missing: {image_path}"
            )

        records.append({
            "image_path": image_path,
            "file_name": file_name,
            "class_name": class_name,
            "class_index": CLASS_TO_INDEX[class_name],
            "source_dataset": "bacterial_viral_dataset",
            "source_class_id": class_id,
        })

    # --------------------------------------------------------
    # B. COVID-19 RADIOGRAPHY DATASET
    # --------------------------------------------------------

    print("Loading COVID images...")

    if not COVID_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"COVID directory not found: {COVID_IMAGE_DIR}"
        )

    for image_path in sorted(COVID_IMAGE_DIR.iterdir()):

        if image_path.is_file() and image_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg"
        }:

            records.append({
                "image_path": image_path,
                "file_name": image_path.name,
                "class_name": "COVID",
                "class_index": CLASS_TO_INDEX["COVID"],
                "source_dataset": "covid19_radiography",
                "source_class_id": None,
            })

    # --------------------------------------------------------
    # C. NORMAL IMAGES FROM COVID DATASET
    # --------------------------------------------------------

    print("Loading additional Normal images...")

    if not NORMAL_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Normal directory not found: {NORMAL_IMAGE_DIR}"
        )

    for image_path in sorted(NORMAL_IMAGE_DIR.iterdir()):

        if image_path.is_file() and image_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg"
        }:

            records.append({
                "image_path": image_path,
                "file_name": image_path.name,
                "class_name": "Normal",
                "class_index": CLASS_TO_INDEX["Normal"],
                "source_dataset": "covid19_radiography",
                "source_class_id": None,
            })

    # --------------------------------------------------------
    # D. VIRAL PNEUMONIA IMAGES FROM COVID DATASET
    # --------------------------------------------------------

    print("Loading additional Viral images...")

    if not VIRAL_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Viral Pneumonia directory not found: {VIRAL_IMAGE_DIR}"
        )

    for image_path in sorted(VIRAL_IMAGE_DIR.iterdir()):

        if image_path.is_file() and image_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg"
        }:

            records.append({
                "image_path": image_path,
                "file_name": image_path.name,
                "class_name": "Viral",
                "class_index": CLASS_TO_INDEX["Viral"],
                "source_dataset": "covid19_radiography",
                "source_class_id": None,
            })

    print("\nImage collection completed.")

    print("Total collected records:", len(records))

    print("\nClass distribution before validation:")

    distribution = pd.Series(
        [record["class_name"] for record in records]
    ).value_counts()

    print(distribution)

    return records


# ============================================================
# 6. VALIDATE, HASH AND DEDUPLICATE IMAGES
# ============================================================

def validate_and_deduplicate_records(records):
    """
    Validate all images and detect exact duplicates globally.

    Rules:
        1. Invalid images are excluded and reported.
        2. SHA256 identifies exact duplicate files.
        3. Same image with different labels causes an error.
        4. Same image with the same label keeps one representative.
        5. Raw images are never modified.
    """

    print("\nStarting image validation and duplicate audit...")

    valid_records = []
    invalid_records = []

    # --------------------------------------------------------
    # A. VALIDATE IMAGES AND CALCULATE HASHES
    # --------------------------------------------------------

    for index, record in enumerate(records, start=1):

        image_path = record["image_path"]

        if not is_valid_image(image_path):

            invalid_records.append({
                "file_name": record["file_name"],
                "image_path": str(image_path),
                "class_name": record["class_name"],
                "source_dataset": record["source_dataset"],
                "reason": "Invalid or unreadable image",
            })

            continue

        image_hash = calculate_sha256(image_path)

        validated_record = record.copy()
        validated_record["sha256"] = image_hash

        valid_records.append(validated_record)

        if index % 500 == 0:
            print(
                f"Validated {index}/{len(records)} records..."
            )

    print("\nImage validation completed.")

    print("Valid images:", len(valid_records))
    print("Invalid images:", len(invalid_records))

    # --------------------------------------------------------
    # B. GROUP IMAGES BY SHA256
    # --------------------------------------------------------

    hash_groups = defaultdict(list)

    for record in valid_records:

        hash_groups[record["sha256"]].append(record)

    unique_records = []
    duplicate_audit = []

    conflicting_groups = []

    # --------------------------------------------------------
    # C. CHECK DUPLICATE GROUPS AND LABEL CONSISTENCY
    # --------------------------------------------------------

    for image_hash, group in hash_groups.items():

        labels = sorted({
            record["class_name"]
            for record in group
        })

        # Same exact image assigned to different classes.
        if len(labels) > 1:

            conflicting_groups.append({
                "sha256": image_hash,
                "class_names": labels,
                "files": [
                    str(record["image_path"])
                    for record in group
                ],
            })

            continue

        # Keep one representative for this exact image.
        representative = sorted(
            group,
            key=lambda record: (
                record["source_dataset"],
                record["file_name"],
            )
        )[0]

        unique_records.append(representative)

        # Record duplicate groups for auditing.
        if len(group) > 1:

            duplicate_audit.append({
                "sha256": image_hash,
                "class_name": labels[0],
                "number_of_copies": len(group),
                "kept_image": str(
                    representative["image_path"]
                ),
                "duplicate_paths": " | ".join(
                    str(record["image_path"])
                    for record in group
                ),
                "source_datasets": ", ".join(
                    sorted({
                        record["source_dataset"]
                        for record in group
                    })
                ),
            })

    # --------------------------------------------------------
    # D. SAVE AUDIT REPORTS
    # --------------------------------------------------------

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(invalid_records).to_csv(
        AUDIT_DIR / "invalid_classification_images.csv",
        index=False
    )

    pd.DataFrame(duplicate_audit).to_csv(
        AUDIT_DIR / "duplicate_classification_images.csv",
        index=False
    )

    pd.DataFrame(conflicting_groups).to_json(
        AUDIT_DIR / "conflicting_classification_labels.json",
        orient="records",
        indent=4
    )

    # --------------------------------------------------------
    # E. STOP IF LABEL CONFLICTS EXIST
    # --------------------------------------------------------

    print("\nDuplicate audit completed.")

    print("Unique image groups:", len(hash_groups))
    print("Duplicate groups:", len(duplicate_audit))
    print("Images retained:", len(unique_records))
    print("Conflicting groups:", len(conflicting_groups))

    if conflicting_groups:

        raise ValueError(
            "Conflicting labels detected for identical images. "
            "Review conflicting_classification_labels.json "
            "before continuing."
        )

    return unique_records


# ============================================================
# 7. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_classification_data(records):
    """
    Split deduplicated records into:
        70% Training
        15% Validation
        15% Testing

    Stratification preserves approximately the same
    class distribution in all three splits.
    """

    print("\nCreating classification dataset splits...")

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No valid records available for splitting.")

    # Check that all four classes are present.
    class_counts = df["class_name"].value_counts()

    print("\nAvailable class counts:")
    print(class_counts)

    missing_classes = set(CLASS_NAMES) - set(class_counts.index)

    if missing_classes:
        raise ValueError(
            f"Missing classification classes: {missing_classes}"
        )

    if (class_counts < 3).any():
        raise ValueError(
            "Each class must have at least 3 images for splitting."
        )

    # --------------------------------------------------------
    # A. SEPARATE TRAINING DATA FROM REMAINING DATA
    # --------------------------------------------------------

    train_df, remaining_df = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["class_name"],
    )

    # --------------------------------------------------------
    # B. SEPARATE VALIDATION AND TEST DATA
    # --------------------------------------------------------

    test_fraction_of_remaining = (
        TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    )

    val_df, test_df = train_test_split(
        remaining_df,
        test_size=test_fraction_of_remaining,
        random_state=RANDOM_SEED,
        stratify=remaining_df["class_name"],
    )

    # --------------------------------------------------------
    # C. ASSIGN SPLIT NAMES
    # --------------------------------------------------------

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Combine all splits.
    final_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # D. VERIFY NO SHA256 HASH CROSSES SPLITS
    # --------------------------------------------------------

    hash_split_counts = (
        final_df.groupby("sha256")["split"].nunique()
    )

    if (hash_split_counts > 1).any():
        raise ValueError(
            "Exact duplicate leakage detected across splits."
        )

    # --------------------------------------------------------
    # E. VERIFY CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nFinal split sizes:")

    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    print("\nClass distribution by split:")

    distribution = pd.crosstab(
        final_df["split"],
        final_df["class_name"],
    )

    distribution = distribution.reindex(
        index=["train", "val", "test"],
        columns=CLASS_NAMES,
        fill_value=0,
    )

    print(distribution)

    return final_df


# ============================================================
# 8. SAVE IMAGES AND GENERATE DATASET MANIFEST
# ============================================================


def save_classification_dataset(final_df):
    """
    Copy images into train, val and test folders.

    Verify copied images using SHA256 and generate:
        - manifest.csv
        - class_distribution.csv
        - split_summary.json

    Original raw images remain unchanged.
    """

    print("\nPreparing classification output directories...")

    # Avoid accidentally mixing results from previous runs.
    if CLASSIFICATION_DATA_DIR.exists():
        existing_files = [
            path for path in CLASSIFICATION_DATA_DIR.rglob("*")
            if path.is_file()
        ]

        if existing_files:
            raise FileExistsError(
                f"Classification output already contains files: "
                f"{CLASSIFICATION_DATA_DIR}. "
                "Move or archive the existing output before rerunning."
            )

    CLASSIFICATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    # --------------------------------------------------------
    # A. COPY IMAGES AND VERIFY THEIR HASHES
    # --------------------------------------------------------

    for _, row in final_df.iterrows():

        split = row["split"]
        class_name = row["class_name"]

        original_path = Path(row["image_path"])

        if not original_path.exists():
            raise FileNotFoundError(
                f"Original image not found: {original_path}"
            )

        destination_dir = CLASSIFICATION_DATA_DIR / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Use the complete SHA256 hash to avoid filename collisions.
        output_filename = f"{row['sha256']}_{original_path.name}"
        destination_path = destination_dir / output_filename

        shutil.copy2(original_path, destination_path)

        # Verify that the copied image is identical to the original.
        copied_hash = calculate_sha256(destination_path)

        if copied_hash != row["sha256"]:
            raise ValueError(
                f"SHA256 verification failed for: {destination_path}"
            )

        if not destination_path.exists():
            raise FileNotFoundError(
                f"Copied image not found: {destination_path}"
            )

        manifest_records.append({
            "split": split,
            "class_name": class_name,
            "class_index": int(row["class_index"]),
            "original_file_name": row["file_name"],
            "original_path": str(original_path),
            "processed_path": str(destination_path),
            "source_dataset": row["source_dataset"],
            "source_class_id": row["source_class_id"],
            "sha256": row["sha256"],
        })

    # --------------------------------------------------------
    # B. VALIDATE MANIFEST PATHS
    # --------------------------------------------------------

    manifest_df = pd.DataFrame(manifest_records)

    missing_paths = [
        path for path in manifest_df["processed_path"]
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest image paths are missing."
        )

    if manifest_df["sha256"].duplicated().any():
        raise ValueError(
            "Duplicate SHA256 hashes found in the final manifest."
        )

    # --------------------------------------------------------
    # C. SAVE MANIFEST
    # --------------------------------------------------------

    manifest_path = CLASSIFICATION_DATA_DIR / "manifest.csv"

    manifest_df.to_csv(manifest_path, index=False)

    # --------------------------------------------------------
    # D. GENERATE CLASS DISTRIBUTION REPORT
    # --------------------------------------------------------

    distribution = pd.crosstab(
        manifest_df["split"],
        manifest_df["class_name"]
    )

    distribution = distribution.reindex(
        index=["train", "val", "test"],
        columns=CLASS_NAMES,
        fill_value=0
    )

    distribution["Total"] = distribution.sum(axis=1)
    distribution.loc["Total"] = distribution.sum(axis=0)

    CLASSIFICATION_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    distribution_path = (
        CLASSIFICATION_RESULTS_DIR / "class_distribution.csv"
    )

    distribution.to_csv(distribution_path)

    # --------------------------------------------------------
    # E. GENERATE SUMMARY JSON
    # --------------------------------------------------------

    summary = {
        "total_images": int(len(manifest_df)),
        "classes": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "random_seed": RANDOM_SEED,
        "split_counts": {
            split: int(
                (manifest_df["split"] == split).sum()
            )
            for split in ["train", "val", "test"]
        },
        "class_distribution": {
            split: {
                class_name: int(
                    distribution.loc[split, class_name]
                )
                for class_name in CLASS_NAMES
            }
            for split in ["train", "val", "test"]
        },
    }

    summary_path = CLASSIFICATION_RESULTS_DIR / "split_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # F. PRINT FINAL SUMMARY
    # --------------------------------------------------------

    print("\nClassification dataset preparation completed.")
    print(f"Verified copied images: {len(manifest_df)}")

    print("\nFinal class distribution:")
    print(distribution)

    print("\nGenerated files:")
    print("Manifest:", manifest_path)
    print("Distribution:", distribution_path)
    print("Summary:", summary_path)

    return manifest_df


# ============================================================
# 9. MAIN EXECUTION
# ============================================================


def main():

    print("=" * 65)
    print("PNEUMONIA CDSS - CLASSIFICATION DATA PREPARATION")
    print("=" * 65)

    # Step 0: Validate split ratios.
    if not abs(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
    ) < 1e-9:
        raise ValueError(
            "Train, validation and test ratios must sum to 1."
        )

    if min(TRAIN_RATIO, VAL_RATIO, TEST_RATIO) <= 0:
        raise ValueError(
            "All split ratios must be greater than zero."
        )

    # Step 1: Load source images and labels.
    records = collect_classification_records()

    if not records:
        raise ValueError("No classification images were found.")

    print(f"\nTotal source records: {len(records)}")

    # Step 2: Validate images and remove exact duplicates.
    unique_records = validate_and_deduplicate_records(records)

    if not unique_records:
        raise ValueError("No valid unique images available.")

    print(f"Valid unique images: {len(unique_records)}")

    # Step 3: Create stratified train/validation/test splits.
    final_df = split_classification_data(unique_records)

    # Step 4: Copy images and generate reports.
    manifest_df = save_classification_dataset(final_df)

    print("\n" + "=" * 65)
    print("DATASET PREPARATION FINISHED SUCCESSFULLY")
    print("Total prepared images:", len(manifest_df))
    print("=" * 65)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/02_prepare_classification_data.py