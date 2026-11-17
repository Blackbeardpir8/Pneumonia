
from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, wilcoxon


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "quantitative_gradcam_analysis"
    / "per_image_attention_statistics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "statistical_gradcam_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 10000
CONFIDENCE_LEVEL = 0.95

GROUPS = [
    "correct_viral",
    "normal_to_covid",
    "viral_to_bacterial",
]

METRICS = [
    "activated_inside_percentage",
    "activated_outside_percentage",
    "mean_activation_inside_lung",
    "mean_activation_outside_lung",
    "lung_area_percentage",
]

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading quantitative Grad-CAM results...")

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "analysis_group",
    "image",
    *METRICS,
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df[df["analysis_group"].isin(GROUPS)].copy()

print("Total images:", len(df))

print("\nGroup counts:")
print(df["analysis_group"].value_counts().to_string())


# ============================================================
# 3. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_mean_ci(
    values,
    iterations=BOOTSTRAP_ITERATIONS,
    confidence=CONFIDENCE_LEVEL,
    seed=RANDOM_STATE,
):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None, None

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(iterations)

    for index in range(iterations):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[index] = np.mean(sample)

    alpha = 1 - confidence

    lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return float(lower), float(upper)


# ============================================================
# 4. CLIFF'S DELTA EFFECT SIZE
# ============================================================

def cliffs_delta(group_a, group_b):

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    # Positive delta means values in A tend to be larger.
    differences = a[:, None] - b[None, :]

    greater = np.sum(differences > 0)
    smaller = np.sum(differences < 0)

    return float(
        (greater - smaller) / (len(a) * len(b))
    )


def interpret_cliffs_delta(delta):

    magnitude = abs(delta)

    if magnitude < 0.147:
        return "negligible"

    if magnitude < 0.33:
        return "small"

    if magnitude < 0.474:
        return "medium"

    return "large"


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\nCalculating descriptive statistics...")

descriptive_rows = []

for group_index, group_name in enumerate(GROUPS):

    group_data = df[
        df["analysis_group"] == group_name
    ]

    for metric_index, metric in enumerate(METRICS):

        values = (
            group_data[metric]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            continue

        lower, upper = bootstrap_mean_ci(
            values,
            seed=(
                RANDOM_STATE
                + group_index * 100
                + metric_index
            ),
        )

        descriptive_rows.append({

            "group": group_name,

            "metric": metric,

            "sample_count": len(values),

            "mean": float(np.mean(values)),

            "median": float(np.median(values)),

            "std": float(
                np.std(values, ddof=1)
            ) if len(values) > 1 else 0.0,

            "minimum": float(np.min(values)),

            "maximum": float(np.max(values)),

            "bootstrap_95ci_lower": lower,

            "bootstrap_95ci_upper": upper,
        })


descriptive_df = pd.DataFrame(
    descriptive_rows
)

descriptive_path = (
    OUTPUT_DIR / "group_descriptive_statistics.csv"
)

descriptive_df.to_csv(
    descriptive_path,
    index=False,
)


# ============================================================
# 6. PAIRWISE MANN-WHITNEY TESTS
# ============================================================

print("\nRunning pairwise Mann-Whitney U tests...")

pairs = [
    ("correct_viral", "normal_to_covid"),
    ("correct_viral", "viral_to_bacterial"),
    ("normal_to_covid", "viral_to_bacterial"),
]

comparison_rows = []

for metric in METRICS:

    for group_a_name, group_b_name in pairs:

        group_a = df.loc[
            df["analysis_group"] == group_a_name,
            metric,
        ].dropna().to_numpy(dtype=float)

        group_b = df.loc[
            df["analysis_group"] == group_b_name,
            metric,
        ].dropna().to_numpy(dtype=float)

        if len(group_a) == 0 or len(group_b) == 0:
            continue

        test_result = mannwhitneyu(
            group_a,
            group_b,
            alternative="two-sided",
            method="auto",
        )

        delta = cliffs_delta(
            group_a,
            group_b,
        )

        comparison_rows.append({

            "metric": metric,

            "group_a": group_a_name,

            "group_b": group_b_name,

            "n_a": len(group_a),

            "n_b": len(group_b),

            "mean_a": float(np.mean(group_a)),

            "mean_b": float(np.mean(group_b)),

            "median_a": float(np.median(group_a)),

            "median_b": float(np.median(group_b)),

            "mean_difference_a_minus_b": float(
                np.mean(group_a) - np.mean(group_b)
            ),

            "mann_whitney_u": float(
                test_result.statistic
            ),

            "p_value": float(
                test_result.pvalue
            ),

            "cliffs_delta": delta,

            "effect_size_interpretation": (
                interpret_cliffs_delta(delta)
            ),
        })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 7. HOLM MULTIPLE-TEST CORRECTION
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    count = len(p_values)

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted_sorted = np.empty(count)

    running_max = 0.0

    for index, p_value in enumerate(sorted_p):

        adjusted = (
            count - index
        ) * p_value

        running_max = max(
            running_max,
            adjusted
        )

        adjusted_sorted[index] = min(
            running_max,
            1.0
        )

    adjusted = np.empty(count)

    adjusted[order] = adjusted_sorted

    return adjusted


if not comparison_df.empty:

    # Correct across all pairwise comparisons and metrics.
    comparison_df["holm_adjusted_p_value"] = (
        holm_correction(
            comparison_df["p_value"].to_numpy()
        )
    )

    comparison_df["significant_after_holm"] = (
        comparison_df["holm_adjusted_p_value"] < 0.05
    )


comparison_path = (
    OUTPUT_DIR / "pairwise_statistical_comparisons.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# 8. WITHIN-IMAGE INSIDE VS OUTSIDE COMPARISON
# ============================================================

print("\nRunning paired inside/outside comparisons...")

paired_rows = []

for group_name in GROUPS:

    group_data = df[
        df["analysis_group"] == group_name
    ].dropna(
        subset=[
            "activated_inside_percentage",
            "activated_outside_percentage",
        ]
    )

    inside = group_data[
        "activated_inside_percentage"
    ].to_numpy(dtype=float)

    outside = group_data[
        "activated_outside_percentage"
    ].to_numpy(dtype=float)

    if len(inside) == 0:
        continue

    if np.allclose(inside, outside):
        statistic = 0.0
        p_value = 1.0

    else:

        paired_result = wilcoxon(
            inside,
            outside,
            alternative="two-sided",
        )

        statistic = float(
            paired_result.statistic
        )

        p_value = float(
            paired_result.pvalue
        )

    differences = outside - inside

    lower, upper = bootstrap_mean_ci(
        differences,
        seed=RANDOM_STATE + 500,
    )

    paired_rows.append({

        "group": group_name,

        "sample_count": len(group_data),

        "mean_inside": float(np.mean(inside)),

        "mean_outside": float(np.mean(outside)),

        "mean_outside_minus_inside": float(
            np.mean(differences)
        ),

        "median_outside_minus_inside": float(
            np.median(differences)
        ),

        "bootstrap_95ci_difference_lower": lower,

        "bootstrap_95ci_difference_upper": upper,

        "wilcoxon_statistic": statistic,

        "p_value": p_value,
    })


paired_df = pd.DataFrame(paired_rows)

if not paired_df.empty:

    paired_df["holm_adjusted_p_value"] = (
        holm_correction(
            paired_df["p_value"].to_numpy()
        )
    )

    paired_df["significant_after_holm"] = (
        paired_df["holm_adjusted_p_value"] < 0.05
    )


paired_path = (
    OUTPUT_DIR / "within_group_paired_comparisons.csv"
)

paired_df.to_csv(
    paired_path,
    index=False,
)


# ============================================================
# 9. SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {

    "experiment": (
        "Statistical Comparison of Grad-CAM Attention"
    ),

    "input_file": str(INPUT_PATH),

    "total_images": len(df),

    "groups": {
        name: int(
            (df["analysis_group"] == name).sum()
        )
        for name in GROUPS
    },

    "metrics": METRICS,

    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,

    "confidence_level": CONFIDENCE_LEVEL,

    "pairwise_test": "Mann-Whitney U, two-sided",

    "multiple_comparison_correction": "Holm",

    "effect_size": "Cliff's delta",

    "within_group_test": "Wilcoxon signed-rank",

    "interpretation_warning": (
        "Exploratory analysis of sampled held-out test images. "
        "Statistical significance does not establish causation, "
        "clinical validity, or disease-specific attention. "
        "The groups differ in source and class composition."
    ),
}

with open(
    OUTPUT_DIR / "statistical_analysis_summary.json",
    "w",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("STATISTICAL GRAD-CAM ANALYSIS COMPLETED")
print("=" * 80)

print("\nDESCRIPTIVE STATISTICS")

print(
    descriptive_df[
        [
            "group",
            "metric",
            "sample_count",
            "mean",
            "median",
            "std",
            "bootstrap_95ci_lower",
            "bootstrap_95ci_upper",
        ]
    ].to_string(index=False)
)

print("\nPAIRWISE COMPARISONS")

print(
    comparison_df[
        [
            "metric",
            "group_a",
            "group_b",
            "mean_difference_a_minus_b",
            "p_value",
            "holm_adjusted_p_value",
            "cliffs_delta",
            "effect_size_interpretation",
            "significant_after_holm",
        ]
    ].to_string(index=False)
)

print("\nWITHIN-GROUP INSIDE VS OUTSIDE")

print(
    paired_df.to_string(index=False)
)

print("\nSaved files:")

print(descriptive_path)
print(comparison_path)
print(paired_path)

print(
    OUTPUT_DIR / "statistical_analysis_summary.json"
)

print("\nAnalysis completed.")

# to run this script, use the following command:
# python scripts/78_statistical_gradcam_comparison.py