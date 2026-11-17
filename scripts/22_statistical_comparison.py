import pandas as pd
from scipy.stats import wilcoxon
import numpy as np

original = pd.read_csv(
    "results/segmentation/validation_metrics.csv"
)

balanced = pd.read_csv(
    "results/segmentation/balanced_validation_metrics.csv"
)

merged = original[["image", "dice"]].merge(
    balanced[["image", "dice"]],
    on="image",
    suffixes=("_original", "_balanced")
)

difference = (
    merged["dice_balanced"] - merged["dice_original"]
)

statistic, p_value = wilcoxon(
    merged["dice_balanced"],
    merged["dice_original"],
    alternative="greater"
)

print("\n===== PAIRED STATISTICAL COMPARISON =====")
print("Number of images:", len(merged))
print("Mean original Dice:", round(merged["dice_original"].mean(), 4))
print("Mean balanced Dice:", round(merged["dice_balanced"].mean(), 4))
print("Mean improvement:", round(difference.mean(), 4))
print("Median improvement:", round(difference.median(), 4))
print("Wilcoxon statistic:", statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Statistically significant improvement.")
else:
    print("Result: No statistically significant improvement.")

# to run 
# python scripts/22_statistical_comparison.py