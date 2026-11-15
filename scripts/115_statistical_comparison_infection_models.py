
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "final_loss_function_test_comparison"
)

ORIGINAL_FILE = RESULTS_DIR / "original_bce_dice_per_image_results.csv"
WEIGHTED_FILE = RESULTS_DIR / "weighted_bce_dice_per_image_results.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "infection_segmentation"
    / "statistical_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PAIRED STATISTICAL COMPARISON")
print("=" * 70)

original = pd.read_csv(ORIGINAL_FILE)
weighted = pd.read_csv(WEIGHTED_FILE)

print("\nOriginal model columns:")
print(original.columns.tolist())

print("\nWeighted model columns:")
print(weighted.columns.tolist())


# ============================================================
# IDENTIFY IMAGE AND DICE COLUMNS
# ============================================================

def find_column(df, candidates, description):
    lookup = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not identify {description} column. "
        f"Available columns: {df.columns.tolist()}"
    )


image_candidates = [
    "image_path",
    "image_name",
    "filename",
    "file_name",
    "image",
    "path",
]

dice_candidates = [
    "dice",
    "dice_score",
    "per_image_dice",
    "mean_dice",
]

original_image_col = find_column(
    original, image_candidates, "image identifier"
)

weighted_image_col = find_column(
    weighted, image_candidates, "image identifier"
)

original_dice_col = find_column(
    original, dice_candidates, "original Dice"
)

weighted_dice_col = find_column(
    weighted, dice_candidates, "weighted Dice"
)


# ============================================================
# PREPARE AND MATCH DATA
# ============================================================

original_data = original[
    [original_image_col, original_dice_col]
].copy()

weighted_data = weighted[
    [weighted_image_col, weighted_dice_col]
].copy()

original_data.columns = ["image_id", "original_dice"]
weighted_data.columns = ["image_id", "weighted_dice"]

# Match images by filename/path, not by CSV row position.
original_data["image_id"] = (
    original_data["image_id"].astype(str).str.replace("\\", "/", regex=False)
)

weighted_data["image_id"] = (
    weighted_data["image_id"].astype(str).str.replace("\\", "/", regex=False)
)

original_data["image_id"] = original_data["image_id"].str.split("/").str[-1]
weighted_data["image_id"] = weighted_data["image_id"].str.split("/").str[-1]

if original_data["image_id"].duplicated().any():
    raise ValueError("Duplicate image identifiers found in original results.")

if weighted_data["image_id"].duplicated().any():
    raise ValueError("Duplicate image identifiers found in weighted results.")

paired = pd.merge(
    original_data,
    weighted_data,
    on="image_id",
    how="inner",
    validate="one_to_one",
)

paired = paired.dropna(
    subset=["original_dice", "weighted_dice"]
).copy()

if len(paired) < 2:
    raise ValueError("Not enough matched images for statistical comparison.")

paired["dice_difference"] = (
    paired["weighted_dice"] - paired["original_dice"]
)

paired.to_csv(
    OUTPUT_DIR / "paired_per_image_dice_results.csv",
    index=False,
)

print(f"\nMatched images: {len(paired)}")

if len(paired) != len(original_data) or len(paired) != len(weighted_data):
    print("WARNING: Not all rows matched. Check image identifiers.")


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

original_scores = paired["original_dice"].to_numpy(dtype=float)
weighted_scores = paired["weighted_dice"].to_numpy(dtype=float)
differences = paired["dice_difference"].to_numpy(dtype=float)

mean_original = float(np.mean(original_scores))
mean_weighted = float(np.mean(weighted_scores))
mean_difference = float(np.mean(differences))
median_difference = float(np.median(differences))
std_difference = float(np.std(differences, ddof=1))

wins = int(np.sum(differences > 0))
losses = int(np.sum(differences < 0))
ties = int(np.sum(differences == 0))


# ============================================================
# PAIRED WILCOXON SIGNED-RANK TEST
# ============================================================

if np.allclose(differences, 0):
    wilcoxon_statistic = 0.0
    p_value = 1.0
else:
    test_result = wilcoxon(
        weighted_scores,
        original_scores,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    wilcoxon_statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)


# ============================================================
# EFFECT SIZE: COHEN'S DZ
# ============================================================

if std_difference > 0:
    cohens_dz = mean_difference / std_difference
else:
    cohens_dz = None


# ============================================================
# PAIRED BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS)

for i in range(BOOTSTRAP_ITERATIONS):
    sampled = rng.choice(
        differences,
        size=len(differences),
        replace=True,
    )

    bootstrap_means[i] = np.mean(sampled)

ci_lower, ci_upper = np.percentile(
    bootstrap_means,
    [2.5, 97.5],
)


# ============================================================
# INTERPRETATION
# ============================================================

if p_value < 0.05:
    significance = "Statistically significant at the 0.05 level"
else:
    significance = "Not statistically significant at the 0.05 level"

if ci_lower > 0:
    ci_interpretation = (
        "The 95% confidence interval is entirely above zero."
    )
elif ci_upper < 0:
    ci_interpretation = (
        "The 95% confidence interval is entirely below zero."
    )
else:
    ci_interpretation = (
        "The 95% confidence interval includes zero."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

summary = {
    "comparison": "Weighted BCE + Dice vs Original BCE + Dice",
    "test_images_matched": int(len(paired)),
    "original_mean_dice": mean_original,
    "weighted_mean_dice": mean_weighted,
    "mean_paired_dice_difference": mean_difference,
    "median_paired_dice_difference": median_difference,
    "std_paired_dice_difference": std_difference,
    "weighted_model_wins": wins,
    "original_model_wins": losses,
    "ties": ties,
    "wilcoxon_statistic": wilcoxon_statistic,
    "wilcoxon_p_value": p_value,
    "significance_at_0_05": significance,
    "cohens_dz": cohens_dz,
    "bootstrap_95_ci_lower": float(ci_lower),
    "bootstrap_95_ci_upper": float(ci_upper),
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "confidence_interval_interpretation": ci_interpretation,
    "threshold_used": 0.50,
    "test_set_used_for_training": False,
    "test_set_used_for_threshold_tuning": False,
}

with open(
    OUTPUT_DIR / "statistical_comparison_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)


summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    OUTPUT_DIR / "statistical_comparison_summary.csv",
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL COMPARISON RESULTS")
print("=" * 70)

print(f"Matched test images:       {len(paired)}")
print(f"Original mean Dice:        {mean_original:.6f}")
print(f"Weighted mean Dice:        {mean_weighted:.6f}")
print(f"Mean Dice improvement:     {mean_difference:.6f}")
print(f"Median Dice improvement:   {median_difference:.6f}")

print("\nPer-image comparison:")
print(f"Weighted model wins:       {wins}")
print(f"Original model wins:       {losses}")
print(f"Ties:                      {ties}")

print("\nStatistical tests:")
print(f"Wilcoxon statistic:        {wilcoxon_statistic:.6f}")
print(f"Wilcoxon p-value:          {p_value:.8f}")
print(f"Significance:              {significance}")

if cohens_dz is not None:
    print(f"Cohen's dz:                {cohens_dz:.6f}")
else:
    print("Cohen's dz:                Undefined (zero difference variance)")

print("\nPaired bootstrap 95% confidence interval:")
print(f"Lower bound:               {ci_lower:.6f}")
print(f"Upper bound:               {ci_upper:.6f}")
print(ci_interpretation)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved:")
print("1. paired_per_image_dice_results.csv")
print("2. statistical_comparison_summary.csv")
print("3. statistical_comparison_summary.json")

print("\nIMPORTANT:")
print("This is a paired comparison of the existing test predictions.")
print("No model training or threshold tuning was performed.")
