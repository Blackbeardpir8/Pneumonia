
"""
SCRIPT 110
Statistical Comparison of Threshold 0.30 vs 0.50
on the independent COVID test set.

Uses paired per-image Dice scores.

Calculates:
- Mean paired Dice difference
- 95% bootstrap confidence interval
- Wilcoxon signed-rank test
- Improved / worsened / unchanged image counts
- Cohen's dz paired effect size

No model inference or training is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "results" / "infection_segmentation"
    / "selected_threshold_independent_test_evaluation"
    / "test_per_image_metrics.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "test_threshold_statistical_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = "BASELINE_0.50"
SELECTED = "VALIDATION_SELECTED_0.30"

STRATEGY = "AUTOMATIC_LUNG_CONSTRAINED"

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42

TOLERANCE = 1e-12


def main():

    print("=" * 80)
    print("TEST-SET PAIRED STATISTICAL COMPARISON")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    df = df[df["strategy"] == STRATEGY].copy()

    baseline = df[
        df["threshold_name"] == BASELINE
    ][["image_name", "dice"]].rename(
        columns={"dice": "baseline_dice"}
    )

    selected = df[
        df["threshold_name"] == SELECTED
    ][["image_name", "dice"]].rename(
        columns={"dice": "selected_dice"}
    )

    paired = baseline.merge(
        selected,
        on="image_name",
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != 583:
        raise ValueError(
            f"Expected 583 paired images, found {len(paired)}"
        )

    paired["dice_difference"] = (
        paired["selected_dice"]
        - paired["baseline_dice"]
    )

    differences = paired["dice_difference"].to_numpy()

    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))
    std_difference = float(np.std(differences, ddof=1))

    improved = int(np.sum(differences > TOLERANCE))
    worsened = int(np.sum(differences < -TOLERANCE))
    unchanged = int(
        len(differences) - improved - worsened
    )

    # --------------------------------------------------------
    # Bootstrap confidence interval for mean paired difference.
    # --------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    bootstrap_means = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    for index in range(BOOTSTRAP_ITERATIONS):

        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_means[index] = np.mean(sample)

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    # --------------------------------------------------------
    # Wilcoxon signed-rank test.
    # --------------------------------------------------------

    nonzero = differences[
        np.abs(differences) > TOLERANCE
    ]

    if len(nonzero):

        test = wilcoxon(
            nonzero,
            alternative="two-sided",
            method="auto",
        )

        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    else:

        statistic = 0.0
        p_value = 1.0

    # --------------------------------------------------------
    # Paired standardized effect size.
    # --------------------------------------------------------

    if std_difference > 0:

        cohens_dz = (
            mean_difference / std_difference
        )

    else:

        cohens_dz = None

    paired.to_csv(
        OUTPUT_DIR / "paired_test_dice_differences.csv",
        index=False,
    )

    summary = {
        "test_images": int(len(paired)),
        "comparison": "Automatic lung-constrained threshold 0.30 vs 0.50",
        "mean_baseline_dice": float(
            paired["baseline_dice"].mean()
        ),
        "mean_selected_dice": float(
            paired["selected_dice"].mean()
        ),
        "mean_paired_dice_difference": mean_difference,
        "median_paired_dice_difference": median_difference,
        "bootstrap_95_percent_ci": {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
            "iterations": BOOTSTRAP_ITERATIONS,
        },
        "improved_images": improved,
        "worsened_images": worsened,
        "unchanged_images": unchanged,
        "wilcoxon_signed_rank": {
            "statistic": statistic,
            "p_value": p_value,
        },
        "cohens_dz": cohens_dz,
        "limitations": [
            "This comparison uses the independent test set.",
            "The threshold was selected using validation data.",
            "Statistical significance does not establish clinical importance.",
            "The results apply to this COVID infection segmentation dataset.",
        ],
    }

    with open(
        OUTPUT_DIR / "test_threshold_statistical_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nPaired test images:", len(paired))

    print("\nMean baseline Dice:", round(
        summary["mean_baseline_dice"], 6
    ))

    print("Mean selected Dice:", round(
        summary["mean_selected_dice"], 6
    ))

    print("Mean paired improvement:", round(
        mean_difference, 6
    ))

    print(
        "95% bootstrap CI:",
        (round(ci_lower, 6), round(ci_upper, 6)),
    )

    print("\nImproved:", improved)
    print("Worsened:", worsened)
    print("Unchanged:", unchanged)

    print("\nWilcoxon statistic:", statistic)
    print("Wilcoxon p-value:", p_value)

    print("Cohen's dz:", cohens_dz)

    print("\n" + "=" * 80)
    print("SCRIPT 110 COMPLETED")
    print("=" * 80)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated:")
    print("1. paired_test_dice_differences.csv")
    print("2. test_threshold_statistical_summary.json")


if __name__ == "__main__":
    main()

# # to run this script, use the following command:
# python scripts/110_statistical_test_threshold_comparison.py