"""
Audit exact duplicate X-ray images in the dedicated
lung segmentation dataset.

This script:
1. Finds X-rays with corresponding masks.
2. Calculates SHA256 hashes of the X-rays.
3. Identifies duplicate images within the segmentation dataset.
4. Checks whether segmentation images occur elsewhere in the raw dataset.
5. Writes CSV reports without modifying raw files.
"""

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import RAW_DATA_DIR, RESULTS_DIR


SEGMENTATION_ROOT = (
    RAW_DATA_DIR
    / "archive (2)"
    / "Lung Segmentation"
)

IMAGE_DIR = SEGMENTATION_ROOT / "CXR_png"
MASK_DIR = SEGMENTATION_ROOT / "masks"

OUTPUT_DIR = RESULTS_DIR / "dataset_audit"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def sha256_file(path):
    """Calculate the SHA256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_images(folder):
    """Return supported image files directly inside a folder."""
    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_csv(path, fieldnames, rows):
    """Write rows to a CSV report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Segmentation image directory not found: {IMAGE_DIR}"
        )

    if not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Segmentation mask directory not found: {MASK_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Match image stems against mask stems.
    masks_by_stem = {
        path.stem.removesuffix("_mask"): path
        for path in find_images(MASK_DIR)
    }

    matched_images = [
        path
        for path in find_images(IMAGE_DIR)
        if path.stem in masks_by_stem
    ]

    print("=" * 65)
    print("SEGMENTATION DATASET DUPLICATE AUDIT")
    print("=" * 65)
    print(f"Segmentation images: {len(find_images(IMAGE_DIR))}")
    print(f"Matched image-mask pairs: {len(matched_images)}")

    # Hash the matched segmentation images.
    segmentation_hashes = defaultdict(list)

    for index, image_path in enumerate(matched_images, start=1):
        digest = sha256_file(image_path)
        segmentation_hashes[digest].append(
            image_path.relative_to(RAW_DATA_DIR).as_posix()
        )

        if index % 100 == 0:
            print(f"Hashed {index}/{len(matched_images)} images")

    internal_duplicates = []

    for digest, paths in segmentation_hashes.items():
        if len(paths) > 1:
            internal_duplicates.append({
                "sha256": digest,
                "number_of_copies": len(paths),
                "relative_paths": " | ".join(paths),
            })

    internal_report = (
        OUTPUT_DIR / "segmentation_internal_duplicates.csv"
    )

    write_csv(
        internal_report,
        ["sha256", "number_of_copies", "relative_paths"],
        internal_duplicates,
    )

    print(
        "\nDuplicate groups within segmentation dataset: "
        f"{len(internal_duplicates)}"
    )

    # Index the matched segmentation hashes.
    target_hashes = set(segmentation_hashes)

    # Scan the other raw datasets for matching image hashes.
    external_matches = []
    scanned_count = 0

    for path in RAW_DATA_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Skip the selected segmentation image directory itself.
        try:
            path.relative_to(IMAGE_DIR)
            continue
        except ValueError:
            pass

        # Ignore the duplicate copy of the dedicated segmentation dataset.
        if (
            "archive (2)" in path.parts
            and "Lung Segmentation" in path.parts
        ):
            continue

        scanned_count += 1

        try:
            digest = sha256_file(path)
        except OSError as error:
            print(f"WARNING: Could not read {path}: {error}")
            continue

        if digest in target_hashes:
            external_matches.append({
                "sha256": digest,
                "segmentation_paths": " | ".join(
                    segmentation_hashes[digest]
                ),
                "other_raw_path": path.relative_to(
                    RAW_DATA_DIR
                ).as_posix(),
            })

        if scanned_count % 1000 == 0:
            print(
                f"Scanned {scanned_count} other image files"
            )

    external_report = (
        OUTPUT_DIR / "segmentation_external_duplicates.csv"
    )

    write_csv(
        external_report,
        [
            "sha256",
            "segmentation_paths",
            "other_raw_path",
        ],
        external_matches,
    )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)
    print(f"Matched segmentation images: {len(matched_images)}")
    print(f"Internal duplicate groups: {len(internal_duplicates)}")
    print(f"External matching image files: {len(external_matches)}")
    print(f"Internal report: {internal_report}")
    print(f"External report: {external_report}")
    print("Raw dataset files were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()



# to run this command 
# python scripts/01c_audit_segmentation_duplicates.py