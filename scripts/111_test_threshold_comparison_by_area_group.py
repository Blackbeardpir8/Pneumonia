
"""
SCRIPT 111
Independent Test-Set Threshold Comparison by Infection Area Group

Compares:
- Baseline threshold: 0.50
- Validation-selected threshold: 0.30

Uses automatically lung-constrained predictions only.

Groups are determined from reference infection-mask area:
- VERY_SMALL: < 1%
- SMALL: 1% to < 10%
- MODERATE: 10% to < 50%
- LARGE: >= 50%

Calculates:
- Mean and median Dice
- Mean precision and recall
- Global pixel Dice, IoU, precision and recall
- Paired Dice improvement
- Bootstrap confidence intervals
- Wilcoxon signed-rank tests

No model training or inference is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

METRICS_CSV = (
    ROOT / "results" / "infection_segmentation"
    / "selected_threshold_independent_test_evaluation"
    / "test_per_image_metrics.csv"
)

REFERENCE_MASK_DIR = (
    ROOT / "data" / "raw" / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Test" / "COVID-19" / "infection masks"
)

OUTPUT_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "test_threshold_area_group_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = "BASELINE_0.50"
SELECTED = "VALIDATION_SELECTED_0.30"

STRATEGY = "AUTOMATIC_LUNG_CONSTRAINED"

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
TOLERANCE = 1e-12

VALID_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ============================================================
# REFERENCE MASK LOOKUP
# ============================================================

def build_lookup(directory):

    if not directory.exists():
        raise FileNotFoundError(directory)

    lookup = {}

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        lookup[path.stem.lower()] = path

    return lookup


def calculate_reference_area(path):

    mask = Image.open(path).convert("L")

    binary = np.asarray(mask) >= 128

    total_pixels = binary.size
    infection_pixels = int(binary.sum())

    return (
        infection_pixels / total_pixels * 100
    )


def assign_area_group(area):

    if area < 1:
        return "VERY_SMALL"

    if area < 10:
        return "SMALL"

    if area < 50:
        return "MODERATE"

    return "LARGE"


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_mean_ci(
    differences,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=RANDOM_SEED,
):

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    if len(differences) == 0:
        return None, None

    if len(differences) == 1:
        value = float(differences[0])
        return value, value

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        iterations,
        dtype=np.float64,
    )

    for index in range(iterations):

        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_means[index] = sample.mean()

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


# ============================================================
# PAIRED STATISTICAL TEST
# ============================================================

def paired_test(group):

    differences = (
        group["selected_dice"].to_numpy()
        - group["baseline_dice"].to_numpy()
    )

    improved = int(
        np.sum(differences > TOLERANCE)
    )

    worsened = int(
        np.sum(differences < -TOLERANCE)
    )

    unchanged = int(
        np.sum(np.abs(differences) <= TOLERANCE)
    )

    mean_difference = float(
        np.mean(differences)
    )

    median_difference = float(
        np.median(differences)
    )

    ci_lower, ci_upper = bootstrap_mean_ci(
        differences,
    )

    nonzero = differences[
        np.abs(differences) > TOLERANCE
    ]

    if len(nonzero) >= 2:

        test = wilcoxon(
            nonzero,
            alternative="two-sided",
            method="auto",
        )

        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    else:

        statistic = None
        p_value = None

    std_difference = (
        float(np.std(differences, ddof=1))
        if len(differences) > 1
        else 0.0
    )

    effect_size = (
        mean_difference / std_difference
        if std_difference > 0
        else None
    )

    return {
        "mean_dice_improvement": mean_difference,
        "median_dice_improvement": median_difference,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "improved_images": improved,
        "worsened_images": worsened,
        "unchanged_images": unchanged,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "cohens_dz": effect_size,
    }


# ============================================================
# GROUP METRICS
# ============================================================

def summarize_strategy(group):

    tp = int(group["tp"].sum())
    fp = int(group["fp"].sum())
    fn = int(group["fn"].sum())

    global_dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 1.0
    )

    global_iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 1.0
    )

    global_precision = (
        tp / (tp + fp)
        if tp + fp else 1.0
    )

    global_recall = (
        tp / (tp + fn)
        if tp + fn else 1.0
    )

    return {
        "image_count": int(len(group)),
        "mean_image_dice": float(group["dice"].mean()),
        "median_image_dice": float(group["dice"].median()),
        "mean_precision": float(group["precision"].mean()),
        "mean_recall": float(group["recall"].mean()),
        "global_pixel_dice": float(global_dice),
        "global_pixel_iou": float(global_iou),
        "global_precision": float(global_precision),
        "global_recall": float(global_recall),
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TEST THRESHOLD COMPARISON BY INFECTION AREA GROUP")
    print("=" * 80)

    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)

    df = pd.read_csv(METRICS_CSV)

    df = df[
        df["strategy"] == STRATEGY
    ].copy()

    baseline = df[
        df["threshold_name"] == BASELINE
    ].copy()

    selected = df[
        df["threshold_name"] == SELECTED
    ].copy()

    if len(baseline) != 583 or len(selected) != 583:
        raise ValueError(
            "Expected 583 records for each threshold."
        )

    baseline = baseline[
        [
            "image_name",
            "dice",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
    ].rename(columns={
        "dice": "baseline_dice",
        "precision": "baseline_precision",
        "recall": "baseline_recall",
        "tp": "baseline_tp",
        "fp": "baseline_fp",
        "fn": "baseline_fn",
        "tn": "baseline_tn",
    })

    selected = selected[
        [
            "image_name",
            "dice",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
    ].rename(columns={
        "dice": "selected_dice",
        "precision": "selected_precision",
        "recall": "selected_recall",
        "tp": "selected_tp",
        "fp": "selected_fp",
        "fn": "selected_fn",
        "tn": "selected_tn",
    })

    paired = baseline.merge(
        selected,
        on="image_name",
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != 583:
        raise ValueError(
            f"Expected 583 paired images, got {len(paired)}"
        )

    # --------------------------------------------------------
    # Assign groups using reference infection-mask area.
    # --------------------------------------------------------

    reference_lookup = build_lookup(
        REFERENCE_MASK_DIR
    )

    print(
        f"\nReference masks discovered: "
        f"{len(reference_lookup)}"
    )

    area_rows = []

    for index, row in paired.iterrows():

        image_name = row["image_name"]
        stem = Path(image_name).stem.lower()

        if stem not in reference_lookup:
            raise FileNotFoundError(
                f"Reference mask missing: {image_name}"
            )

        area = calculate_reference_area(
            reference_lookup[stem]
        )

        area_rows.append({
            "image_name": image_name,
            "reference_area_percentage": area,
            "area_group": assign_area_group(area),
        })

        if (index + 1) % 100 == 0:
            print(
                f"Analyzed reference areas: "
                f"{index + 1}/583"
            )

    area_df = pd.DataFrame(area_rows)

    paired = paired.merge(
        area_df,
        on="image_name",
        how="inner",
        validate="one_to_one",
    )

    paired["dice_difference"] = (
        paired["selected_dice"]
        - paired["baseline_dice"]
    )

    paired.to_csv(
        OUTPUT_DIR / "paired_test_results_with_area_groups.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Group comparison.
    # --------------------------------------------------------

    group_rows = []
    statistical_rows = []

    group_order = [
        "VERY_SMALL",
        "SMALL",
        "MODERATE",
        "LARGE",
    ]

    for area_group in group_order:

        group = paired[
            paired["area_group"] == area_group
        ].copy()

        if group.empty:
            continue

        baseline_metrics = summarize_strategy(
            group.rename(columns={
                "baseline_dice": "dice",
                "baseline_precision": "precision",
                "baseline_recall": "recall",
                "baseline_tp": "tp",
                "baseline_fp": "fp",
                "baseline_fn": "fn",
            })
        )

        selected_metrics = summarize_strategy(
            group.rename(columns={
                "selected_dice": "dice",
                "selected_precision": "precision",
                "selected_recall": "recall",
                "selected_tp": "tp",
                "selected_fp": "fp",
                "selected_fn": "fn",
            })
        )

        group_rows.append({
            "area_group": area_group,
            "reference_area_mean": float(
                group["reference_area_percentage"].mean()
            ),
            "threshold": 0.50,
            **{
                f"baseline_{key}": value
                for key, value in baseline_metrics.items()
            },
            "selected_threshold": 0.30,
            **{
                f"selected_{key}": value
                for key, value in selected_metrics.items()
            },
        })

        stats = paired_test(group)

        statistical_rows.append({
            "area_group": area_group,
            "image_count": int(len(group)),
            **stats,
        })

    group_df = pd.DataFrame(group_rows)
    stats_df = pd.DataFrame(statistical_rows)

    group_df.to_csv(
        OUTPUT_DIR / "test_area_group_threshold_metrics.csv",
        index=False,
    )

    stats_df.to_csv(
        OUTPUT_DIR / "test_area_group_paired_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Overall group counts.
    # --------------------------------------------------------

    counts = (
        paired["area_group"]
        .value_counts()
        .reindex(group_order, fill_value=0)
        .to_dict()
    )

    # --------------------------------------------------------
    # Summary JSON.
    # --------------------------------------------------------

    summary = {
        "test_images": int(len(paired)),
        "baseline_threshold": 0.50,
        "selected_threshold": 0.30,
        "strategy": STRATEGY,
        "area_group_counts": {
            key: int(value)
            for key, value in counts.items()
        },
        "group_metrics": group_rows,
        "paired_statistics": statistical_rows,
        "group_definitions": {
            "VERY_SMALL": "Reference infection area < 1%",
            "SMALL": "Reference infection area >= 1% and < 10%",
            "MODERATE": "Reference infection area >= 10% and < 50%",
            "LARGE": "Reference infection area >= 50%",
        },
        "limitations": [
            "Groups are based on reference annotation area.",
            "Groups are geometric categories, not clinical severity.",
            "Small groups may have limited statistical power.",
            "The threshold was selected using validation data.",
            "The independent test set was used only for evaluation.",
            "Statistical significance does not establish clinical benefit.",
        ],
    }

    with open(
        OUTPUT_DIR / "test_area_group_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Display.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST IMAGE COUNTS BY AREA GROUP")
    print("=" * 80)

    print(pd.Series(counts).to_string())

    print("\n" + "=" * 80)
    print("THRESHOLD METRICS BY AREA GROUP")
    print("=" * 80)

    print(
        group_df[
            [
                "area_group",
                "baseline_image_count",
                "baseline_mean_image_dice",
                "selected_mean_image_dice",
                "baseline_mean_precision",
                "selected_mean_precision",
                "baseline_mean_recall",
                "selected_mean_recall",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL COMPARISON")
    print("=" * 80)

    print(stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("SCRIPT 111 COMPLETED")
    print("=" * 80)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. paired_test_results_with_area_groups.csv")
    print("2. test_area_group_threshold_metrics.csv")
    print("3. test_area_group_paired_statistics.csv")
    print("4. test_area_group_analysis_summary.json")


if __name__ == "__main__":
    main()

# # to run this script, use the following command:
# python scripts/111_test_threshold_comparison_by_area_group.py