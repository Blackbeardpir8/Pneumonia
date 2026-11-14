
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "data/raw/archive(4)/Infection Segmentation Data/"
    "Infection Segmentation Data"
)

OUTPUT = ROOT / "results/infection_mask_investigation"
OUTPUT.mkdir(parents=True, exist_ok=True)

records = []

for split in ["Train", "Val", "Test"]:

    base = DATASET / split / "COVID-19"

    infection_dir = base / "infection masks"
    lung_dir = base / "lung masks"

    for infection_path in infection_dir.glob("*.png"):

        lung_path = lung_dir / infection_path.name

        if not lung_path.exists():
            continue

        infection = np.array(
            Image.open(infection_path).convert("L")
        ) > 0

        lung = np.array(
            Image.open(lung_path).convert("L")
        ) > 0

        intersection = np.count_nonzero(infection & lung)
        infection_area = np.count_nonzero(infection)
        lung_area = np.count_nonzero(lung)
        union = np.count_nonzero(infection | lung)

        records.append({
            "split": split,
            "filename": infection_path.stem,
            "infection_area": infection_area,
            "lung_area": lung_area,
            "infection_percentage": (
                infection_area / infection.size * 100
            ),
            "lung_percentage": (
                lung_area / lung.size * 100
            ),
            "overlap_percentage_of_infection": (
                intersection / infection_area * 100
                if infection_area else 0
            ),
            "iou_infection_vs_lung": (
                intersection / union if union else 0
            ),
            "identical_masks": np.array_equal(infection, lung),
        })

df = pd.DataFrame(records)

print("\n========== COVID MASK OVERLAP AUDIT ==========\n")

print("Total samples:", len(df))
print("Identical infection/lung masks:", df["identical_masks"].sum())

print("\nOverlap percentage of infection:")
print(df["overlap_percentage_of_infection"].describe().round(3))

print("\nInfection vs lung IoU:")
print(df["iou_infection_vs_lung"].describe().round(4))

print("\nIdentical masks by split:")
print(df.groupby("split")["identical_masks"].sum())

print("\nExamples with identical masks:")
print(
    df[df["identical_masks"]][
        ["split", "filename", "infection_percentage"]
    ].head(20).to_string(index=False)
)

output_path = OUTPUT / "infection_lung_overlap_audit.csv"
df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
