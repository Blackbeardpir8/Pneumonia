
import pandas as pd
from pathlib import Path

files = {
    "Original Validation": "results/segmentation/validation_metrics.csv",
    "Balanced Validation": "results/segmentation/balanced_validation_metrics.csv",
    "Balanced Test": "results/segmentation/final_test_metrics.csv",
}

results = []

for experiment, filepath in files.items():

    df = pd.read_csv(filepath)

    # Derive source from image filename when needed
    if "source" not in df.columns:
        df["source"] = df["image"].str.split("_").str[0]

    # Overall results
    groups = [("Overall", df)]

    # Source-wise results
    groups.extend(df.groupby("source"))

    for source, group in groups:

        results.append({
            "experiment": experiment,
            "source": source,
            "images": len(group),
            "mean_dice": group["dice"].mean(),
            "median_dice": group["dice"].median(),
            "dice_std": group["dice"].std(ddof=1),
            "mean_iou": group["iou"].mean(),
            "median_iou": group["iou"].median(),
            "iou_std": group["iou"].std(ddof=1),
        })

summary = pd.DataFrame(results)

output = Path(
    "results/segmentation/segmentation_results_summary.csv"
)

summary.to_csv(output, index=False)

print("\n========== SEGMENTATION RESULTS SUMMARY ==========\n")
print(summary.round(4).to_string(index=False))

print("\nSaved:", output.resolve())
