
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "results/infection_mask_investigation/"
    "infection_area_quality_audit.csv"
)

OUTPUT = INPUT.parent

df = pd.read_csv(INPUT)

# Infection area relative to lung area.
df["infection_to_lung_percentage"] = (
    df["infection_to_lung_percentage"]
    if "infection_to_lung_percentage" in df.columns
    else df["infection_area"] / df["lung_area"] * 100
)

def classify(row):
    ratio = row["infection_to_lung_percentage"]

    if row["identical_masks"]:
        return "IDENTICAL_TO_LUNG"

    if ratio < 1:
        return "VERY_SMALL"

    if ratio < 10:
        return "SMALL"

    if ratio < 50:
        return "MODERATE"

    if ratio < 90:
        return "LARGE"

    return "VERY_LARGE"

df["quality_group"] = df.apply(classify, axis=1)

print("\n========== INFECTION MASK QUALITY GROUPS ==========\n")

summary = df.groupby("quality_group").agg(
    samples=("filename", "count"),
    mean_infection_area=(
        "infection_to_lung_percentage", "mean"
    ),
    minimum_area=(
        "infection_to_lung_percentage", "min"
    ),
    maximum_area=(
        "infection_to_lung_percentage", "max"
    ),
).round(3)

print(summary)

print("\n========== QUALITY GROUPS BY SPLIT ==========\n")

print(
    pd.crosstab(
        df["split"],
        df["quality_group"]
    )
)

output_path = OUTPUT / "infection_mask_quality_groups.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
