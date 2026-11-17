
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "results/infection_mask_investigation/"
    "infection_segmentation_audit.csv"
)

df = pd.read_csv(CSV_PATH)

print("\n========== INFECTION MASK STATISTICS ==========\n")

for class_name in ["COVID-19", "Non-COVID", "Normal"]:

    subset = df[df["class"] == class_name]

    total = len(subset)
    empty = int(subset["infection_mask_empty"].sum())
    non_empty = total - empty

    print(f"\nCLASS: {class_name}")
    print("-" * 50)
    print("Total samples:", total)
    print("Non-empty masks:", non_empty)
    print("Empty masks:", empty)

    if non_empty:
        values = subset.loc[
            ~subset["infection_mask_empty"],
            "infection_foreground_percent"
        ]

        print("Minimum foreground %:", round(values.min(), 3))
        print("Median foreground %:", round(values.median(), 3))
        print("Mean foreground %:", round(values.mean(), 3))
        print("Maximum foreground %:", round(values.max(), 3))

    print(
        "Dimension mismatches:",
        int((~subset["image_mask_dimensions_match"]).sum())
    )

print("\n========== SPLIT-WISE SUMMARY ==========\n")

summary = df.groupby(["split", "class"]).agg(
    samples=("filename", "count"),
    empty_masks=("infection_mask_empty", "sum"),
    mean_foreground=("infection_foreground_percent", "mean"),
).round(3)

print(summary)

output = CSV_PATH.parent / "infection_mask_statistics.csv"
summary.to_csv(output)

print("\nStatistics saved:", output)

# to run this script, use the following command:
# python scripts/84_analyze_infection_mask_statistics.py