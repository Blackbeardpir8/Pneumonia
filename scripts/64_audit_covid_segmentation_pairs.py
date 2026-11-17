
import csv
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path("data/raw/archive/COVID-19_Radiography_Dataset")

CLASSES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity",
]

OUTPUT_DIR = Path("results/segmentation/covid_mask_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for class_name in CLASSES:

    image_dir = BASE_DIR / class_name / "images"
    mask_dir = BASE_DIR / class_name / "masks"

    image_files = {
        p.name: p for p in image_dir.iterdir()
        if p.is_file()
    }

    mask_files = {
        p.name: p for p in mask_dir.iterdir()
        if p.is_file()
    }

    matched_names = sorted(
        set(image_files) & set(mask_files)
    )

    missing_masks = sorted(
        set(image_files) - set(mask_files)
    )

    missing_images = sorted(
        set(mask_files) - set(image_files)
    )

    print(f"\nChecking {class_name}")
    print("Images:", len(image_files))
    print("Masks:", len(mask_files))
    print("Matched:", len(matched_names))

    for filename in matched_names:

        image_path = image_files[filename]
        mask_path = mask_files[filename]

        try:
            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(mask_path) as mask:
                mask_array = np.array(mask.convert("L"))

            unique_values = np.unique(mask_array)

            foreground_pixels = int(
                np.count_nonzero(mask_array)
            )

            rows.append({
                "class_name": class_name,
                "filename": filename,
                "image_width": image_size[0],
                "image_height": image_size[1],
                "mask_width": mask_array.shape[1],
                "mask_height": mask_array.shape[0],
                "dimensions_match": (
                    image_size == (
                        mask_array.shape[1],
                        mask_array.shape[0]
                    )
                ),
                "mask_unique_values": ",".join(
                    map(str, unique_values.tolist())
                ),
                "foreground_pixels": foreground_pixels,
                "mask_empty": foreground_pixels == 0,
                "status": "OK",
            })

        except Exception as error:
            rows.append({
                "class_name": class_name,
                "filename": filename,
                "status": f"ERROR: {error}",
            })

    print("Missing masks:", len(missing_masks))
    print("Missing images:", len(missing_images))

output_file = OUTPUT_DIR / "covid_segmentation_pair_audit.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nAudit completed.")
print("Report:", output_file)
print("Total records:", len(rows))

# to run this script, use the following command:
# python scripts/64_audit_covid_segmentation_pairs.py