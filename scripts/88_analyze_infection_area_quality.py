
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "results/infection_mask_investigation/"
    "infection_lung_overlap_audit.csv"
)

OUTPUT = CSV_PATH.parent

df = pd.read_csv(CSV_PATH)

# Infection area as a percentage of lung area.
df["infection_to_lung_percentage"] = (
    df["infection_area"] / df["lung_area"] * 100
)

print("\n========== INFECTION AREA QUALITY ANALYSIS ==========\n")

print("Total COVID samples:", len(df))

print("\nInfection area relative to lung area:")
print(
    df["infection_to_lung_percentage"]
    .describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
    .round(3)
)

print("\n========== IDENTICAL MASK ANALYSIS ==========\n")

identical = df[df["identical_masks"] == True]
different = df[df["identical_masks"] == False]

print("Identical masks:", len(identical))
print("Different masks:", len(different))

print(
    "Identical masks percentage:",
    round(len(identical) / len(df) * 100, 2)
)

print("\n========== IDENTICAL MASKS BY SPLIT ==========\n")

print(
    df.groupby("split").agg(
        total=("filename", "count"),
        identical=("identical_masks", "sum"),
        mean_infection_to_lung=(
            "infection_to_lung_percentage", "mean"
        ),
    ).round(3)
)

print("\n========== EXTREME ANNOTATIONS ==========\n")

print("\nHighest infection-to-lung ratios:")
print(
    df.nlargest(10, "infection_to_lung_percentage")[
        ["split", "filename", "infection_to_lung_percentage",
         "identical_masks"]
    ].to_string(index=False)
)

print("\nLowest infection-to-lung ratios:")
print(
    df.nsmallest(10, "infection_to_lung_percentage")[
        ["split", "filename", "infection_to_lung_percentage",
         "identical_masks"]
    ].to_string(index=False)
)

# Save enriched report.
output_path = OUTPUT / "infection_area_quality_audit.csv"
df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
