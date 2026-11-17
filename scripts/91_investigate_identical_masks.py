
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "data/raw/archive(4)/Infection Segmentation Data/"
    "Infection Segmentation Data"
)

CSV_PATH = (
    ROOT / "results/infection_mask_investigation/"
    "infection_area_quality_audit.csv"
)

OUTPUT = CSV_PATH.parent

df = pd.read_csv(CSV_PATH)
identical = df[df["identical_masks"] == True].copy()

records = []

for _, row in identical.iterrows():

    base = DATASET / row["split"] / "COVID-19"

    filename = row["filename"]

    image_path = base / "images" / f"{filename}.png"
    infection_path = base / "infection masks" / f"{filename}.png"
    lung_path = base / "lung masks" / f"{filename}.png"

    image = np.array(
        Image.open(image_path).convert("L")
    )

    infection = np.array(
        Image.open(infection_path).convert("L")
    ) > 0

    lung = np.array(
        Image.open(lung_path).convert("L")
    ) > 0

    records.append({
        "split": row["split"],
        "filename": filename,
        "image_mean_intensity": image.mean(),
        "image_std_intensity": image.std(),
        "infection_pixels": infection.sum(),
        "lung_pixels": lung.sum(),
        "image_shape": str(image.shape),
        "infection_equals_lung": np.array_equal(
            infection, lung
        ),
    })

result = pd.DataFrame(records)

print("\n========== IDENTICAL MASK INVESTIGATION ==========\n")

print("Total identical masks:", len(result))
print(
    "Still identical after direct image loading:",
    result["infection_equals_lung"].sum()
)

print("\nSplit distribution:")
print(result["split"].value_counts())

print("\nImage intensity statistics:")
print(
    result[
        ["image_mean_intensity", "image_std_intensity"]
    ].describe().round(3)
)

print("\nFirst 15 samples:")
print(result.head(15).to_string(index=False))

output_path = OUTPUT / "identical_mask_investigation.csv"
result.to_csv(output_path, index=False)

print("\nSaved:", output_path)

# to run this script, use the following command:
# python scripts/91_investigate_identical_masks.py