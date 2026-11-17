import pandas as pd

# Load evaluation results
original = pd.read_csv(
    "results/segmentation/validation_metrics.csv"
)

balanced = pd.read_csv(
    "results/segmentation/balanced_validation_metrics.csv"
)

# Sort both datasets
original = original.sort_values("image").reset_index(drop=True)
balanced = balanced.sort_values("image").reset_index(drop=True)

# Basic verification
print("\n========== DATASET VERIFICATION ==========")

print("Original rows:", len(original))
print("Balanced rows:", len(balanced))

print(
    "Same validation images:",
    original["image"].equals(balanced["image"])
)

print(
    "Same thresholds:",
    (original["threshold"] == balanced["threshold"]).all()
)

print("Original thresholds:", original["threshold"].unique())
print("Balanced thresholds:", balanced["threshold"].unique())

# Source-wise performance
original["source"] = original["image"].str.split("_").str[0]
balanced["source"] = balanced["image"].str.split("_").str[0]

print("\n========== ORIGINAL MODEL ==========")

print(
    original.groupby("source")["dice"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(4)
)

print("\n========== BALANCED MODEL ==========")

print(
    balanced.groupby("source")["dice"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(4)
)

# Image-by-image comparison
merged = original[
    ["image", "dice"]
].merge(
    balanced[["image", "dice"]],
    on="image",
    suffixes=("_original", "_balanced")
)

merged["delta"] = (
    merged["dice_balanced"] - merged["dice_original"]
)

print("\n========== IMAGE-BY-IMAGE IMPROVEMENT ==========")

print(merged["delta"].describe().round(4))

print("\n10 images with lowest improvement:")

print(
    merged.sort_values("delta").head(10).to_string(index=False)
)

print("\n10 images with highest improvement:")

print(
    merged.sort_values("delta", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Save comparison
merged.to_csv(
    "results/segmentation/original_vs_balanced_comparison.csv",
    index=False
)

print("\nComparison saved successfully.")

# to run this script, use the following command:
# python scripts/21_compare_original_and_balanced.py