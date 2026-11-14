"""
Script 01: Inspect and audit the raw pneumonia datasets.

This initial audit:
1. Counts image files in each directory.
2. Identifies existing train, val, and test directories.
3. Checks whether images can be opened.
4. Finds exact duplicate image files using SHA-256 hashes.
5. Checks potential train/test leakage in recognized classification splits.
6. Checks image-mask filename matching for the segmentation datasets.
7. Saves reports under results/dataset_audit/.

IMPORTANT:
- Never modifies files in data/raw/.
- Does not randomly split any dataset.
- Does not assign ambiguous labels.
- Does not copy images into data/processed/ yet.
  Dataset mappings must be reviewed before preparation.
"""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import sys
from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. LOAD PROJECT PATHS AND SETTINGS
# ============================================================

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import RAW_DATA_DIR, RESULTS_DIR


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

AUDIT_DIR = RESULTS_DIR / "dataset_audit"

# macOS metadata files are not actual dataset images.
EXCLUDED_PARTS = {
    "__MACOSX",
    ".ipynb_checkpoints",
}

# These are the candidate image/mask directories observed in the
# actual folder tree. We will not treat both repeated directory
# trees as independent datasets.
SEGMENTATION_CANDIDATES = [
    (
        Path("archive (2)/Lung Segmentation/CXR_png"),
        Path("archive (2)/Lung Segmentation/masks"),
    ),
    (
        Path("archive/COVID-19_Radiography_Dataset"),
        None,
    ),
]


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def is_image_file(path):
    """Return True when the file has a recognized image extension."""
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part in EXCLUDED_PARTS for part in path.parts)
    )


def get_images(directory):
    """Return image files recursively beneath a directory."""
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if is_image_file(path)
    )


def sha256_file(path):
    """Calculate a file's SHA-256 hash in small memory-efficient blocks."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def check_image(path):
    """
    Verify that an image can be opened and decoded.

    Returns:
        (True, "") if valid
        (False, error message) otherwise
    """
    try:
        with Image.open(path) as image:
            image.verify()

        return True, ""

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ) as exc:
        return False, str(exc)


def write_csv(path, fieldnames, rows):
    """Write a CSV report, including its header."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. DIRECTORY INVENTORY
# ============================================================

def inspect_directories():
    """
    Count images in directories that directly contain image files.

    The report includes relative paths, so no absolute machine path
    needs to be embedded in the report.
    """
    rows = []

    directories = sorted(
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_dir()
        and not any(part in EXCLUDED_PARTS for part in path.parts)
    )

    for directory in directories:
        direct_images = [
            path
            for path in directory.iterdir()
            if is_image_file(path)
        ]

        if not direct_images:
            continue

        rows.append({
            "relative_folder": str(
                directory.relative_to(RAW_DATA_DIR)
            ),
            "direct_image_count": len(direct_images),
            "sample_filenames": " | ".join(
                path.name for path in direct_images[:5]
            ),
        })

    write_csv(
        AUDIT_DIR / "directory_inventory.csv",
        [
            "relative_folder",
            "direct_image_count",
            "sample_filenames",
        ],
        rows,
    )

    print(f"Directories containing images: {len(rows)}")
    print("Saved: directory_inventory.csv")


# ============================================================
# 4. IMAGE VALIDITY AND EXACT DUPLICATE AUDIT
# ============================================================

def inspect_images():
    """
    Verify image files and calculate exact file hashes.

    Identical hashes indicate byte-for-byte identical files.
    Different hashes do not prove that images show different patients.
    """
    images = get_images(RAW_DATA_DIR)

    hash_groups = defaultdict(list)
    invalid_rows = []

    print(f"\nImage files discovered: {len(images)}")
    print("Checking image validity and calculating SHA-256 hashes...")

    for index, path in enumerate(images, start=1):
        if index % 1000 == 0:
            print(f"Processed {index}/{len(images)} images")

        relative_path = str(path.relative_to(RAW_DATA_DIR))

        valid, error = check_image(path)

        if not valid:
            invalid_rows.append({
                "relative_path": relative_path,
                "error": error,
            })
            continue

        try:
            digest = sha256_file(path)
            hash_groups[digest].append(relative_path)

        except OSError as exc:
            invalid_rows.append({
                "relative_path": relative_path,
                "error": f"Hashing error: {exc}",
            })

    duplicate_rows = []

    for digest, paths in hash_groups.items():
        if len(paths) > 1:
            duplicate_rows.append({
                "sha256": digest,
                "number_of_copies": len(paths),
                "relative_paths": " | ".join(paths),
            })

    write_csv(
        AUDIT_DIR / "invalid_images.csv",
        ["relative_path", "error"],
        invalid_rows,
    )

    write_csv(
        AUDIT_DIR / "exact_duplicate_images.csv",
        ["sha256", "number_of_copies", "relative_paths"],
        duplicate_rows,
    )

    print(f"Invalid or unreadable images: {len(invalid_rows)}")
    print(f"Exact-duplicate hash groups: {len(duplicate_rows)}")
    print("Saved: invalid_images.csv")
    print("Saved: exact_duplicate_images.csv")


# ============================================================
# 5. CLASSIFICATION SPLIT AUDIT
# ============================================================

def inspect_classification_splits():
    """
    Summarize recognized split directories and identify exact image
    hashes shared between train and test.

    This is a preliminary audit. It does not establish patient-level
    independence, and it does not assign new class labels.
    """
    split_counts = Counter()
    split_hashes = defaultdict(set)
    hash_to_splits = defaultdict(set)

    split_names = {"train", "val", "test"}

    for dataset_root in sorted(RAW_DATA_DIR.iterdir()):
        if not dataset_root.is_dir():
            continue

        for directory in dataset_root.rglob("*"):
            if (
                not directory.is_dir()
                or directory.name.lower() not in split_names
                or any(
                    part in EXCLUDED_PARTS
                    for part in directory.parts
                )
            ):
                continue

            split_name = directory.name.lower()

            # Ignore directories that do not contain images.
            images = get_images(directory)

            if not images:
                continue

            dataset_relative = str(
                directory.relative_to(RAW_DATA_DIR)
            )

            split_counts[(dataset_relative, split_name)] = len(images)

            for path in images:
                try:
                    digest = sha256_file(path)
                except OSError:
                    continue

                split_hashes[
                    (dataset_relative, split_name)
                ].add(digest)

                # Record hashes by dataset family and split.
                # This allows comparisons within the same top-level
                # archive without comparing unrelated datasets.
                family = path.relative_to(RAW_DATA_DIR).parts[0]
                hash_to_splits[(family, digest)].add(split_name)

    count_rows = [
        {
            "split_folder": folder,
            "image_count": count,
        }
        for (folder, _), count in sorted(split_counts.items())
    ]

    write_csv(
        AUDIT_DIR / "classification_split_counts.csv",
        ["split_folder", "image_count"],
        count_rows,
    )

    leakage_rows = []

    for (family, digest), splits in hash_to_splits.items():
        if "train" in splits and "test" in splits:
            leakage_rows.append({
                "dataset_family": family,
                "sha256": digest,
                "splits": ", ".join(sorted(splits)),
            })

    write_csv(
        AUDIT_DIR / "potential_split_leakage.csv",
        ["dataset_family", "sha256", "splits"],
        leakage_rows,
    )

    print("\nClassification split folders inspected.")
    print(f"Exact hashes shared between train and test: {len(leakage_rows)}")
    print("Saved: classification_split_counts.csv")
    print("Saved: potential_split_leakage.csv")


# ============================================================
# 6. SEGMENTATION IMAGE-MASK AUDIT
# ============================================================

def inspect_segmentation_pairs():
    """
    Check image-mask filename correspondence for the known
    segmentation dataset candidates.

    The COVID-19 Radiography dataset stores images and masks inside
    separate subdirectories for each category.
    """
    rows = []

    for image_relative, mask_relative in SEGMENTATION_CANDIDATES:
        image_root = RAW_DATA_DIR / image_relative

        if mask_relative is None:
            # Inspect the COVID-19 Radiography image/mask subfolders.
            for category in [
                "COVID",
                "Lung_Opacity",
                "Normal",
                "Viral Pneumonia",
            ]:
                image_dir = image_root / category / "images"
                mask_dir = image_root / category / "masks"

                if not image_dir.is_dir() or not mask_dir.is_dir():
                    continue

                images = get_images(image_dir)
                masks = get_images(mask_dir)

                image_names = {p.name for p in images}
                mask_names = {p.name for p in masks}

                rows.append({
                    "dataset": str(image_relative),
                    "category": category,
                    "image_count": len(images),
                    "mask_count": len(masks),
                    "matched_image_mask_pairs": len(
                        image_names & mask_names
                    ),
                    "images_without_mask": len(
                        image_names - mask_names
                    ),
                    "masks_without_image": len(
                        mask_names - image_names
                    ),
                    "test_image_count": 0,
                    "test_images_in_cxr_png": 0,
                    "test_images_with_masks": 0,
                })

            continue

        # Dedicated Lung Segmentation dataset
        mask_root = RAW_DATA_DIR / mask_relative

        if not image_root.is_dir() or not mask_root.is_dir():
            continue

        images = get_images(image_root)
        masks = get_images(mask_root)

        # Correct matching using stem (without extension)
        # Masks: CHNCXR_0001_0_mask.png  →  stem "CHNCXR_0001_0_mask"
        # After removesuffix → "CHNCXR_0001_0"
        # Images: CHNCXR_0001_0.png      →  stem "CHNCXR_0001_0"
        image_stems = {path.stem for path in images}
        mask_stems = {
            path.stem.removesuffix("_mask")
            for path in masks
        }

        matched_stems = image_stems & mask_stems

        # Optional: also look at the test folder if it exists
        test_dir = image_root.parent / "test"
        test_images = get_images(test_dir) if test_dir.is_dir() else []
        test_stems = {path.stem for path in test_images}

        test_in_images = test_stems & image_stems
        test_with_masks = test_stems & mask_stems

        rows.append({
            "dataset": str(image_relative),
            "category": "lung_segmentation",
            "image_count": len(images),
            "mask_count": len(masks),
            "matched_image_mask_pairs": len(matched_stems),
            "images_without_mask": len(
                image_stems - mask_stems
            ),
            "masks_without_image": len(
                mask_stems - image_stems
            ),
            "test_image_count": len(test_images),
            "test_images_in_cxr_png": len(test_in_images),
            "test_images_with_masks": len(test_with_masks),
        })

    # Save the corrected report.
    write_csv(
        AUDIT_DIR / "segmentation_mask_audit.csv",
        [
            "dataset",
            "category",
            "image_count",
            "mask_count",
            "matched_image_mask_pairs",
            "images_without_mask",
            "masks_without_image",
            "test_image_count",
            "test_images_in_cxr_png",
            "test_images_with_masks",
        ],
        rows,
    )

    print("\nSegmentation image-mask audit completed.")
    print("Saved: segmentation_mask_audit.csv")

    for row in rows:
        if row["category"] == "lung_segmentation":
            print("\nDedicated Lung Segmentation dataset:")
            print(f"Images: {row['image_count']}")
            print(f"Masks: {row['mask_count']}")
            print(
                "Matched image-mask pairs:",
                row["matched_image_mask_pairs"],
            )
            print(
                "Images without masks:",
                row["images_without_mask"],
            )
            print(
                "Masks without images:",
                row["masks_without_image"],
            )
            print("Test images:", row["test_image_count"])
            print(
                "Test images also in CXR_png:",
                row["test_images_in_cxr_png"],
            )
            print(
                "Test images with matching masks:",
                row["test_images_with_masks"],
            )


# ============================================================
# 7. MAIN
# ============================================================

def main():
    """Run the initial read-only dataset audit."""
    if not RAW_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DATA_DIR}"
        )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PNEUMONIA CDSS — INITIAL DATASET AUDIT")
    print("=" * 65)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Audit output: {AUDIT_DIR}")

    inspect_directories()
    inspect_images()
    inspect_classification_splits()
    inspect_segmentation_pairs()

    print("\n" + "=" * 65)
    print("INITIAL AUDIT COMPLETED")
    print("Raw datasets were not modified.")
    print("No images were copied into processed/.")
    print("=" * 65)


if __name__ == "__main__":
    main()



# to run this script in terminal command prompt,
# (venv) F:\Hemlata Thesis\Code> python scripts/01_inspect_and_prepare_data.py