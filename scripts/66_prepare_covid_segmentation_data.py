
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()

CLASSIFICATION_MANIFEST = (
    PROJECT_ROOT /
    "data/processed/classification/manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "data/processed/segmentation_covid"
)

IMAGE_SIZE = (224, 224)

ALLOWED_CLASSES = [
    "COVID",
    "Normal",
    "Viral",
]

CLASS_MAPPING = {
    "COVID": "COVID",
    "Normal": "Normal",
    "Viral": "Viral Pneumonia",
}

SPLITS = ["train", "val", "test"]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def prepare_image(image_path, output_path):

    with Image.open(image_path) as image:

        image = image.convert("L")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image.save(output_path)


def prepare_mask(mask_path, output_path):

    with Image.open(mask_path) as mask:

        mask = mask.convert("L")

        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        mask_array = np.array(mask)

        binary_mask = np.where(
            mask_array > 0,
            255,
            0
        ).astype(np.uint8)

        Image.fromarray(binary_mask).save(output_path)


# --------------------------------------------------
# LOAD CLASSIFICATION MANIFEST
# --------------------------------------------------

if not CLASSIFICATION_MANIFEST.exists():
    raise FileNotFoundError(CLASSIFICATION_MANIFEST)

df = pd.read_csv(CLASSIFICATION_MANIFEST)

df = df[
    (df["source_dataset"] == "covid19_radiography") &
    (df["class_name"].isin(ALLOWED_CLASSES))
].copy()

print("\nSelected source records:", len(df))

print("\nSplit distribution:")
print(df["split"].value_counts().to_string())

# --------------------------------------------------
# PREPARE DATASET
# --------------------------------------------------

prepared_records = []
failed_records = []

for split in SPLITS:

    split_df = df[df["split"] == split]

    image_output_dir = OUTPUT_ROOT / split / "images"
    mask_output_dir = OUTPUT_ROOT / split / "masks"

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nPreparing {split}: {len(split_df)} images")

    for index, row in split_df.iterrows():

        filename = row["original_file_name"]

        image_path = Path(row["original_path"])

        original_class = CLASS_MAPPING[row["class_name"]]

        mask_path = (
            image_path.parent.parent /
            "masks" /
            filename
        )

        if not image_path.exists():
            failed_records.append({
                "filename": filename,
                "split": split,
                "reason": "Image not found",
            })
            continue

        if not mask_path.exists():
            failed_records.append({
                "filename": filename,
                "split": split,
                "reason": "Mask not found",
            })
            continue

        output_name = (
            f"{row['sha256']}_{Path(filename).stem}.png"
        )

        image_output = image_output_dir / output_name
        mask_output = mask_output_dir / output_name

        try:

            prepare_image(
                image_path,
                image_output
            )

            prepare_mask(
                mask_path,
                mask_output
            )

            prepared_records.append({
                "split": split,
                "class_name": row["class_name"],
                "original_file_name": filename,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path),
                "processed_image_path": str(image_output),
                "processed_mask_path": str(mask_output),
                "sha256": row["sha256"],
            })

        except Exception as error:

            failed_records.append({
                "filename": filename,
                "split": split,
                "reason": str(error),
            })

    print(
        f"Completed {split}: "
        f"{sum(r['split'] == split for r in prepared_records)}"
    )


# --------------------------------------------------
# SAVE MANIFEST AND FAILURES
# --------------------------------------------------

manifest_path = OUTPUT_ROOT / "manifest.csv"

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=prepared_records[0].keys()
    )

    writer.writeheader()
    writer.writerows(prepared_records)


if failed_records:

    failure_path = OUTPUT_ROOT / "processing_failures.csv"

    pd.DataFrame(failed_records).to_csv(
        failure_path,
        index=False
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("COVID SEGMENTATION DATA PREPARATION SUMMARY")
print("=" * 55)

print("Expected records:", len(df))
print("Prepared records:", len(prepared_records))
print("Failed records:", len(failed_records))

print("\nPrepared split counts:")

summary = pd.DataFrame(prepared_records)

print(
    summary.groupby(
        ["split", "class_name"]
    ).size().to_string()
)

print("\nManifest:", manifest_path)

print("\nDataset preparation completed.")

# to run this script, use the following command:
# python scripts/66_prepare_covid_segmentation_data.py