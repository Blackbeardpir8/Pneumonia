
"""
SCRIPT 105
Evaluate Selective Automatic Lung Constraint

Compares:
A. Unconstrained infection segmentation
B. Automatic lung constraint on every image
C. Automatic lung constraint only when predicted lung-mask
   quality checks pass.

Quality checks use predicted lung masks only.
Reference lung masks are NOT used to decide whether to
apply the constraint.

This is an exploratory validation experiment.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "results" / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
    / "automatic_lung_constraint_per_image.csv"
)

MASK_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
    / "predicted_lung_masks"
)

OUTPUT_DIR = (
    ROOT / "results" / "infection_segmentation"
    / "selective_lung_constraint_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exploratory mask-quality thresholds.
# These are NOT clinically validated thresholds.
MIN_LUNG_AREA_RATIO = 0.10
MAX_LUNG_AREA_RATIO = 0.60
MIN_LARGEST_COMPONENT_RATIO = 0.30
MIN_COMPONENT_AREA_RATIO = 0.005
MAX_SIGNIFICANT_COMPONENTS = 3

DICE_TOLERANCE = 1e-6


# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_ALIASES = {
    "image_name": ["image_name", "filename", "image"],
    "group": [
        "annotation_group",
        "infection_group",
        "area_group",
        "subgroup",
        "group",
    ],
    "unconstrained_dice": ["unconstrained_dice"],
    "constrained_dice": ["predicted_lung_constrained_dice"],
    "unconstrained_fp": ["unconstrained_fp"],
    "constrained_fp": ["predicted_lung_constrained_fp"],
    "unconstrained_fn": ["unconstrained_fn"],
    "constrained_fn": ["predicted_lung_constrained_fn"],
}


def find_column(df, logical_name, required=True):

    for name in COLUMN_ALIASES[logical_name]:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"Missing column: {logical_name}\n"
            f"Accepted names: {COLUMN_ALIASES[logical_name]}\n"
            f"Available: {list(df.columns)}"
        )

    return None


# ============================================================
# MASK FILE LOOKUP
# ============================================================


def build_mask_lookup():

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Predicted lung mask directory not found:\n{MASK_DIR}"
        )

    lookup = {}

    for path in MASK_DIR.rglob("*"):

        if path.is_file() and path.suffix.lower() in [
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        ]:

            stem = path.stem.lower()

            # Remove the suffix used by Script 102.
            if stem.endswith("_lung_prediction"):
                stem = stem[:-len("_lung_prediction")]

            lookup[stem] = path

    return lookup



def find_mask(image_name, lookup):

    stem = Path(str(image_name)).stem.lower()

    if stem in lookup:
        return lookup[stem]

    raise FileNotFoundError(
        f"No predicted lung mask found for {image_name}\n"
        f"Expected matching filename stem: {stem}"
    )


# ============================================================
# PREDICTED LUNG MASK QUALITY CHECKS
# ============================================================

def inspect_lung_mask(mask_path):

    mask = np.asarray(
        Image.open(mask_path).convert("L")
    )

    binary = mask > 127

    total_pixels = binary.size
    foreground_pixels = int(binary.sum())

    area_ratio = foreground_pixels / total_pixels

    if foreground_pixels == 0:

        return {
            "lung_area_ratio": 0.0,
            "significant_components": 0,
            "largest_component_ratio": 0.0,
            "quality_pass": False,
            "quality_reason": "EMPTY_MASK",
        }

    # Find connected foreground components.
    labels, component_count = ndimage.label(binary)

    component_sizes = np.bincount(labels.ravel())[1:]

    min_component_pixels = max(
        1,
        int(total_pixels * MIN_COMPONENT_AREA_RATIO)
    )

    significant_sizes = component_sizes[
        component_sizes >= min_component_pixels
    ]

    significant_count = len(significant_sizes)

    largest_component_ratio = (
        float(significant_sizes.max() / foreground_pixels)
        if significant_count > 0
        else 0.0
    )

    # Apply independent mask-quality checks.
    area_ok = (
        MIN_LUNG_AREA_RATIO
        <= area_ratio
        <= MAX_LUNG_AREA_RATIO
    )

    components_ok = (
        1 <= significant_count <= MAX_SIGNIFICANT_COMPONENTS
    )

    largest_component_ok = (
        largest_component_ratio >= MIN_LARGEST_COMPONENT_RATIO
    )

    quality_pass = (
        area_ok
        and components_ok
        and largest_component_ok
    )

    failed_checks = []

    if not area_ok:
        failed_checks.append("AREA_OUTSIDE_RANGE")

    if not components_ok:
        failed_checks.append("COMPONENT_COUNT_OUTSIDE_RANGE")

    if not largest_component_ok:
        failed_checks.append("LARGEST_COMPONENT_TOO_SMALL")

    reason = (
        "PASS"
        if quality_pass
        else ";".join(failed_checks)
    )

    return {
        "lung_area_ratio": float(area_ratio),
        "significant_components": int(significant_count),
        "largest_component_ratio": float(largest_component_ratio),
        "quality_pass": bool(quality_pass),
        "quality_reason": reason,
    }


# ============================================================
# STRATEGY COMPARISON
# ============================================================

def calculate_strategy_metrics(df, strategy_name, dice_col):

    dice = df[dice_col].to_numpy(dtype=float)

    return {
        "strategy": strategy_name,
        "image_count": int(len(df)),
        "mean_dice": float(np.mean(dice)),
        "median_dice": float(np.median(dice)),
        "std_dice": float(np.std(dice, ddof=1)),
    }


def calculate_wilcoxon(before, after):

    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)

    difference = after - before

    if np.all(np.abs(difference) <= DICE_TOLERANCE):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "interpretation": "No measurable paired differences.",
        }

    statistic, p_value = wilcoxon(
        before,
        after,
        alternative="two-sided",
        zero_method="wilcox",
    )

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "interpretation": (
            "Statistically significant paired difference."
            if p_value < 0.05
            else "No statistically significant paired difference."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SELECTIVE AUTOMATIC LUNG CONSTRAINT EVALUATION")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    raw = pd.read_csv(INPUT_CSV)

    print(f"\nInput records: {len(raw)}")

    image_col = find_column(raw, "image_name")
    group_col = find_column(raw, "group", required=False)

    required_columns = {
        "unconstrained_dice": find_column(raw, "unconstrained_dice"),
        "constrained_dice": find_column(raw, "constrained_dice"),
        "unconstrained_fp": find_column(raw, "unconstrained_fp"),
        "constrained_fp": find_column(raw, "constrained_fp"),
        "unconstrained_fn": find_column(raw, "unconstrained_fn"),
        "constrained_fn": find_column(raw, "constrained_fn"),
    }

    df = pd.DataFrame()

    df["image_name"] = raw[image_col].astype(str)

    df["annotation_group"] = (
        raw[group_col].astype(str)
        if group_col
        else "UNKNOWN"
    )

    for logical_name, source_col in required_columns.items():
        df[logical_name] = pd.to_numeric(
            raw[source_col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "unconstrained_dice",
            "constrained_dice",
            "unconstrained_fp",
            "constrained_fp",
            "unconstrained_fn",
            "constrained_fn",
        ]
    ).copy()

    print(f"Valid metric records: {len(df)}")

    # --------------------------------------------------------
    # Inspect all predicted lung masks.
    # --------------------------------------------------------

    print("\nBuilding predicted lung-mask lookup...")

    mask_lookup = build_mask_lookup()

    print(f"Predicted masks discovered: {len(mask_lookup)}")

    quality_rows = []

    for index, row in df.iterrows():

        mask_path = find_mask(
            row["image_name"],
            mask_lookup,
        )

        quality = inspect_lung_mask(mask_path)

        quality_rows.append({
            "image_name": row["image_name"],
            **quality,
        })

        if (len(quality_rows) % 100 == 0):
            print(
                f"Inspected {len(quality_rows)}/{len(df)} masks"
            )

    quality_df = pd.DataFrame(quality_rows)

    df = df.merge(
        quality_df,
        on="image_name",
        how="left",
        validate="one_to_one",
    )

    if df["quality_pass"].isna().any():
        raise ValueError(
            "Some images did not receive a lung-mask quality result."
        )

    df["quality_pass"] = df["quality_pass"].astype(bool)

    # --------------------------------------------------------
    # Build three strategies.
    # --------------------------------------------------------

    df["selective_dice"] = np.where(
        df["quality_pass"],
        df["constrained_dice"],
        df["unconstrained_dice"],
    )

    df["selective_fp"] = np.where(
        df["quality_pass"],
        df["constrained_fp"],
        df["unconstrained_fp"],
    )

    df["selective_fn"] = np.where(
        df["quality_pass"],
        df["constrained_fn"],
        df["unconstrained_fn"],
    )

    df["selective_strategy"] = np.where(
        df["quality_pass"],
        "CONSTRAINED",
        "UNCONSTRAINED",
    )

    # --------------------------------------------------------
    # Quality-check coverage.
    # --------------------------------------------------------

    passed = int(df["quality_pass"].sum())
    failed = int((~df["quality_pass"]).sum())

    coverage = passed / len(df) * 100

    print("\n" + "=" * 80)
    print("PREDICTED LUNG MASK QUALITY")
    print("=" * 80)

    print(f"Passed quality checks: {passed}")
    print(f"Failed quality checks: {failed}")
    print(f"Constraint coverage: {coverage:.2f}%")

    print("\nQuality failure reasons:")

    print(
        df.loc[~df["quality_pass"], "quality_reason"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Strategy metrics.
    # --------------------------------------------------------

    strategy_rows = [
        calculate_strategy_metrics(
            df,
            "A_UNCONSTRAINED",
            "unconstrained_dice",
        ),
        calculate_strategy_metrics(
            df,
            "B_ALWAYS_CONSTRAINED",
            "constrained_dice",
        ),
        calculate_strategy_metrics(
            df,
            "C_SELECTIVE_CONSTRAINT",
            "selective_dice",
        ),
    ]

    strategy_df = pd.DataFrame(strategy_rows)

    strategy_df.to_csv(
        OUTPUT_DIR / "strategy_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    print(strategy_df.to_string(index=False))

    # --------------------------------------------------------
    # Pixel-level comparison.
    # --------------------------------------------------------

    pixel_rows = []

    for strategy, fp_col, fn_col in [
        (
            "A_UNCONSTRAINED",
            "unconstrained_fp",
            "unconstrained_fn",
        ),
        (
            "B_ALWAYS_CONSTRAINED",
            "constrained_fp",
            "constrained_fn",
        ),
        (
            "C_SELECTIVE_CONSTRAINT",
            "selective_fp",
            "selective_fn",
        ),
    ]:

        pixel_rows.append({
            "strategy": strategy,
            "total_false_positives": int(df[fp_col].sum()),
            "total_false_negatives": int(df[fn_col].sum()),
            "mean_fp_per_image": float(df[fp_col].mean()),
            "mean_fn_per_image": float(df[fn_col].mean()),
        })

    pixel_df = pd.DataFrame(pixel_rows)

    pixel_df.to_csv(
        OUTPUT_DIR / "strategy_pixel_comparison.csv",
        index=False,
    )

    print("\nPIXEL-LEVEL COMPARISON")
    print(pixel_df.to_string(index=False))

    # --------------------------------------------------------
    # Paired statistical tests.
    # --------------------------------------------------------

    tests = {
        "selective_vs_unconstrained": calculate_wilcoxon(
            df["unconstrained_dice"],
            df["selective_dice"],
        ),
        "selective_vs_always_constrained": calculate_wilcoxon(
            df["constrained_dice"],
            df["selective_dice"],
        ),
        "always_constrained_vs_unconstrained": calculate_wilcoxon(
            df["unconstrained_dice"],
            df["constrained_dice"],
        ),
    }

    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL TESTS")
    print("=" * 80)

    for name, result in tests.items():
        print(f"\n{name}")
        print(result)

    # --------------------------------------------------------
    # Annotation-group analysis.
    # --------------------------------------------------------

    group_rows = []

    for group_name, group_df in df.groupby(
        "annotation_group",
        observed=True,
    ):

        group_rows.append({
            "annotation_group": str(group_name),
            "image_count": int(len(group_df)),
            "quality_pass_count": int(
                group_df["quality_pass"].sum()
            ),
            "quality_pass_percentage": float(
                group_df["quality_pass"].mean() * 100
            ),
            "unconstrained_mean_dice": float(
                group_df["unconstrained_dice"].mean()
            ),
            "always_constrained_mean_dice": float(
                group_df["constrained_dice"].mean()
            ),
            "selective_mean_dice": float(
                group_df["selective_dice"].mean()
            ),
        })

    group_df = pd.DataFrame(group_rows)

    group_df.to_csv(
        OUTPUT_DIR / "annotation_group_strategy_comparison.csv",
        index=False,
    )

    print("\nANNOTATION-GROUP COMPARISON")
    print(group_df.to_string(index=False))

    # --------------------------------------------------------
    # Save per-image analysis.
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_DIR / "selective_constraint_per_image.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save summary JSON.
    # --------------------------------------------------------

    summary = {
        "study": "Selective automatic lung constraint evaluation",
        "image_count": int(len(df)),
        "quality_thresholds": {
            "minimum_lung_area_ratio": MIN_LUNG_AREA_RATIO,
            "maximum_lung_area_ratio": MAX_LUNG_AREA_RATIO,
            "minimum_largest_component_ratio":
                MIN_LARGEST_COMPONENT_RATIO,
            "minimum_component_area_ratio":
                MIN_COMPONENT_AREA_RATIO,
            "maximum_significant_components":
                MAX_SIGNIFICANT_COMPONENTS,
        },
        "quality_check_results": {
            "passed": passed,
            "failed": failed,
            "coverage_percentage": float(coverage),
        },
        "strategies": strategy_rows,
        "pixel_comparison": pixel_rows,
        "paired_tests": tests,
        "annotation_group_results": group_rows,
        "limitations": [
            "Quality thresholds are exploratory heuristics.",
            "Predicted lung masks alone determine constraint selection.",
            "Reference infection masks are used only for evaluation.",
            "Validation results do not establish generalization.",
            "Annotation groups represent mask geometry, not clinical severity.",
            "No clinical diagnosis or severity claim is made.",
        ],
    }

    with open(
        OUTPUT_DIR / "selective_constraint_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4, allow_nan=False)

    print("\n" + "=" * 80)
    print("SCRIPT 105 COMPLETED")
    print("=" * 80)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. strategy_comparison.csv")
    print("2. strategy_pixel_comparison.csv")
    print("3. annotation_group_strategy_comparison.csv")
    print("4. selective_constraint_per_image.csv")
    print("5. selective_constraint_summary.json")


if __name__ == "__main__":
    main()
