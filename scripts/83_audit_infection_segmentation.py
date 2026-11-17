
from pathlib import Path
from PIL import Image
import numpy as np
import csv

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "data/raw/archive(4)/Infection Segmentation Data/"
    "Infection Segmentation Data"
)

OUTPUT = ROOT / "results/infection_mask_investigation"
OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["Train", "Val", "Test"]
CLASSES = ["COVID-19", "Non-COVID", "Normal"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

records = []

for split in SPLITS:
    for class_name in CLASSES:

        base = DATASET / split / class_name

        image_dir = base / "images"
        infection_dir = base / "infection masks"
        lung_dir = base / "lung masks"

        images = {
            p.stem: p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        }

        infection_masks = {
            p.stem: p for p in infection_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        }

        lung_masks = {
            p.stem: p for p in lung_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        }

        image_names = set(images)
        infection_names = set(infection_masks)
        lung_names = set(lung_masks)

        matched = image_names & infection_names & lung_names

        print(f"\n{'=' * 60}")
        print(f"{split} | {class_name}")
        print(f"{'=' * 60}")
        print("Images:", len(images))
        print("Infection masks:", len(infection_masks))
        print("Lung masks:", len(lung_masks))
        print("Complete filename matches:", len(matched))
        print("Images without infection mask:", len(image_names - infection_names))
        print("Infection masks without image:", len(infection_names - image_names))

        for stem in sorted(matched):
            image_path = images[stem]
            infection_path = infection_masks[stem]
            lung_path = lung_masks[stem]

            image = np.array(Image.open(image_path).convert("L"))
            infection = np.array(Image.open(infection_path).convert("L"))
            lung = np.array(Image.open(lung_path).convert("L"))

            infection_binary = infection > 0
            lung_binary = lung > 0

            infection_pixels = np.count_nonzero(infection_binary)
            lung_pixels = np.count_nonzero(lung_binary)

            outside_lung = np.count_nonzero(
                infection_binary & ~lung_binary
            )

            records.append({
                "split": split,
                "class": class_name,
                "filename": stem,
                "image_height": image.shape[0],
                "image_width": image.shape[1],
                "infection_mask_height": infection.shape[0],
                "infection_mask_width": infection.shape[1],
                "lung_mask_height": lung.shape[0],
                "lung_mask_width": lung.shape[1],
                "infection_unique_values": str(np.unique(infection).tolist()),
                "infection_foreground_pixels": infection_pixels,
                "infection_foreground_percent": (
                    infection_pixels / infection.size * 100
                ),
                "lung_foreground_percent": (
                    lung_pixels / lung.size * 100
                ),
                "infection_outside_lung_pixels": outside_lung,
                "infection_mask_empty": infection_pixels == 0,
                "image_mask_dimensions_match": (
                    image.shape == infection.shape
                ),
            })

report_path = OUTPUT / "infection_segmentation_audit.csv"

if records:
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

print("\n" + "=" * 60)
print("AUDIT COMPLETED")
print("=" * 60)
print("Complete matched samples:", len(records))
print("CSV report:", report_path)

if records:
    print("Empty infection masks:", sum(r["infection_mask_empty"] for r in records))
    print("Dimension mismatches:", sum(not r["image_mask_dimensions_match"] for r in records))

# to run this script, use the following command:
# python scripts/83_audit_infection_segmentation.py