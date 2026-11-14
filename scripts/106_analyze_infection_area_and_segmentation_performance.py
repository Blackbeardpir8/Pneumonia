
"""
SCRIPT 106
Infection Area and Segmentation Performance Analysis

Analyzes the relationship between reference infection-mask area
and infection segmentation performance across 466 validation images.

Compares:
1. Unconstrained infection predictions
2. Automatically lung-constrained infection predictions

Uses:
- Script 102 per-image validation metrics
- COVID-QU-Ex reference infection masks

No model training is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, mannwhitneyu


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

METRICS_CSV = (
    ROOT
    / "results"
    / "infection_segmentation"
    / "automatic_lung_constraint_analysis"
    / "automatic_lung_constraint_per_image.csv"
)

INFECTION_MASK_DIR = (
    ROOT
    / "data"
    / "raw"
    / "archive(4)"
    / "Infection Segmentation Data"
    / "Infection Segmentation Data"
    / "Val"
    / "COVID-19"
    / "infection masks"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "infection_segmentation"
    / "infection_area_performance_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference infection area is measured as a percentage of
# the complete 256x256 image.
AREA_BINS = [
    0,
    1,
    10,
    50,
    90,
    100.000001,
]

AREA_LABELS = [
    "VERY_SMALL",
    "SMALL",
    "MODERATE",
    "LARGE",
    "VERY_LARGE",
]


# ============================================================
# HELPERS
# ============================================================

def find_column(df, name, required=True):

    aliases = {
        "image_name": ["image_name", "filename", "image"],
        "annotation_group": ["annotation_group"],
        "unconstrained_fp": ["unconstrained_fp"],
        "unconstrained_fn": ["unconstrained_fn"],
        "constrained_fp": ["predicted_lung_constrained_fp"],
        "constrained_fn": ["predicted_lung_constrained_fn"],
        "unconstrained_dice": ["unconstrained_dice"],
        "constrained_dice": ["predicted_lung_constrained_dice"],
    }

    for candidate in aliases[name]:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"Missing required column: {name}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def build_mask_lookup():

    if not INFECTION_MASK_DIR.exists():
        raise FileNotFoundError(
            f"Infection mask directory not found:\n"
            f"{INFECTION_MASK_DIR}"
        )

    lookup = {}

    for path in INFECTION_MASK_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in [
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        ]:
            continue

        lookup[path.stem.lower()] = path

    return lookup


def find_reference_mask(image_name, lookup):

    stem = Path(str(image_name)).stem.lower()

    if stem not in lookup:
        raise FileNotFoundError(
            f"Reference infection mask not found for: {image_name}"
        )

    return lookup[stem]


def load_binary_mask(path):

    mask = np.asarray(
        Image.open(path).convert("L")
    )

    return mask >= 128


def calculate_metrics(reference_area, fp, fn):

    # Reference positive pixels = TP + FN.
    tp = int(reference_area - fn)

    if tp < 0:
        raise ValueError(
            "Calculated true positives are negative. "
            "Check reference mask and metric alignment."
        )

    predicted_area = int(tp + fp)

    dice_denominator = (
        2 * tp + fp + fn
    )

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    iou_denominator = tp + fp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 1.0
    )

    area_ratio = (
        predicted_area / reference_area
        if reference_area > 0
        else None
    )

    return {
        "tp": tp,
        "fp": int(fp),
        "fn": int(fn),
        "predicted_area": predicted_area,
        "dice_recalculated": float(dice),
        "iou_recalculated": float(iou),
        "precision_recalculated": float(precision),
        "recall_recalculated": float(recall),
        "predicted_reference_area_ratio": (
            float(area_ratio)
            if area_ratio is not None
            else None
        ),
    }


def safe_spearman(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return {
            "sample_count": int(len(x)),
            "correlation": None,
            "p_value": None,
            "interpretation": "Insufficient samples.",
        }

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return {
            "sample_count": int(len(x)),
            "correlation": None,
            "p_value": None,
            "interpretation": "Insufficient variation.",
        }

    result = spearmanr(x, y)

    return {
        "sample_count": int(len(x)),
        "correlation": float(result.statistic),
        "p_value": float(result.pvalue),
        "interpretation": (
            "Positive association"
            if result.statistic > 0
            else "Negative association"
            if result.statistic < 0
            else "No rank association"
        ),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 80)
    print("INFECTION AREA AND SEGMENTATION PERFORMANCE ANALYSIS")
    print("=" * 80)

    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)

    raw = pd.read_csv(METRICS_CSV)

    print(f"\nMetrics records: {len(raw)}")

    columns = {
        name: find_column(raw, name)
        for name in [
            "image_name",
            "unconstrained_fp",
            "unconstrained_fn",
            "constrained_fp",
            "constrained_fn",
            "unconstrained_dice",
            "constrained_dice",
        ]
    }

    group_col = find_column(
        raw,
        "annotation_group",
        required=False,
    )

    print("\nBuilding reference infection-mask lookup...")

    mask_lookup = build_mask_lookup()

    print(f"Reference infection masks discovered: {len(mask_lookup)}")

    # --------------------------------------------------------
    # Calculate reference infection area for each image.
    # --------------------------------------------------------

    rows = []

    for index, source_row in raw.iterrows():

        image_name = str(
            source_row[columns["image_name"]]
        )

        mask_path = find_reference_mask(
            image_name,
            mask_lookup,
        )

        reference_mask = load_binary_mask(mask_path)

        reference_area = int(reference_mask.sum())
        total_pixels = int(reference_mask.size)

        reference_area_percentage = (
            reference_area / total_pixels * 100
        )

        unconstrained = calculate_metrics(
            reference_area,
            int(source_row[columns["unconstrained_fp"]]),
            int(source_row[columns["unconstrained_fn"]]),
        )

        constrained = calculate_metrics(
            reference_area,
            int(source_row[columns["constrained_fp"]]),
            int(source_row[columns["constrained_fn"]]),
        )

        rows.append({
            "image_name": image_name,
            "reference_infection_area_pixels": reference_area,
            "total_image_pixels": total_pixels,
            "reference_infection_area_percentage":
                reference_area_percentage,
            "annotation_group": (
                str(source_row[group_col])
                if group_col
                else "UNKNOWN"
            ),

            "unconstrained_reported_dice": float(
                source_row[columns["unconstrained_dice"]]
            ),
            "unconstrained_dice": unconstrained[
                "dice_recalculated"
            ],
            "unconstrained_iou": unconstrained[
                "iou_recalculated"
            ],
            "unconstrained_precision": unconstrained[
                "precision_recalculated"
            ],
            "unconstrained_recall": unconstrained[
                "recall_recalculated"
            ],
            "unconstrained_tp": unconstrained["tp"],
            "unconstrained_fp": unconstrained["fp"],
            "unconstrained_fn": unconstrained["fn"],
            "unconstrained_predicted_area":
                unconstrained["predicted_area"],
            "unconstrained_area_ratio":
                unconstrained["predicted_reference_area_ratio"],

            "constrained_reported_dice": float(
                source_row[columns["constrained_dice"]]
            ),
            "constrained_dice": constrained[
                "dice_recalculated"
            ],
            "constrained_iou": constrained[
                "iou_recalculated"
            ],
            "constrained_precision": constrained[
                "precision_recalculated"
            ],
            "constrained_recall": constrained[
                "recall_recalculated"
            ],
            "constrained_tp": constrained["tp"],
            "constrained_fp": constrained["fp"],
            "constrained_fn": constrained["fn"],
            "constrained_predicted_area":
                constrained["predicted_area"],
            "constrained_area_ratio":
                constrained["predicted_reference_area_ratio"],
        })

        if (len(rows) % 100 == 0):
            print(f"Processed {len(rows)}/{len(raw)} images")

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Validate reconstructed Dice against original CSV.
    # --------------------------------------------------------

    max_unconstrained_difference = float(
        (
            df["unconstrained_dice"]
            - df["unconstrained_reported_dice"]
        ).abs().max()
    )

    max_constrained_difference = float(
        (
            df["constrained_dice"]
            - df["constrained_reported_dice"]
        ).abs().max()
    )

    print("\nDice reconstruction verification:")
    print(
        f"Maximum unconstrained difference: "
        f"{max_unconstrained_difference:.10f}"
    )
    print(
        f"Maximum constrained difference: "
        f"{max_constrained_difference:.10f}"
    )

    if (
        max_unconstrained_difference > 1e-5
        or max_constrained_difference > 1e-5
    ):
        raise ValueError(
            "Reconstructed Dice does not match Script 102 metrics. "
            "Stop and investigate before interpreting the results."
        )

    # --------------------------------------------------------
    # Create area groups from actual reference masks.
    # --------------------------------------------------------

    df["reference_area_group"] = pd.cut(
        df["reference_infection_area_percentage"],
        bins=AREA_BINS,
        labels=AREA_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)

    # --------------------------------------------------------
    # Per-image performance change.
    # --------------------------------------------------------

    df["dice_change"] = (
        df["constrained_dice"]
        - df["unconstrained_dice"]
    )

    df["unconstrained_area_error_percentage"] = (
        df["unconstrained_area_ratio"] - 1
    ) * 100

    df["constrained_area_error_percentage"] = (
        df["constrained_area_ratio"] - 1
    ) * 100

    df.to_csv(
        OUTPUT_DIR / "infection_area_per_image_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Area-group performance.
    # --------------------------------------------------------

    group_rows = []

    for group_name, group_df in df.groupby(
        "reference_area_group",
        observed=True,
    ):

        group_rows.append({
            "reference_area_group": str(group_name),
            "image_count": int(len(group_df)),
            "mean_reference_area_percentage": float(
                group_df["reference_infection_area_percentage"].mean()
            ),
            "median_reference_area_percentage": float(
                group_df["reference_infection_area_percentage"].median()
            ),
            "unconstrained_mean_dice": float(
                group_df["unconstrained_dice"].mean()
            ),
            "constrained_mean_dice": float(
                group_df["constrained_dice"].mean()
            ),
            "unconstrained_mean_precision": float(
                group_df["unconstrained_precision"].mean()
            ),
            "unconstrained_mean_recall": float(
                group_df["unconstrained_recall"].mean()
            ),
            "constrained_mean_precision": float(
                group_df["constrained_precision"].mean()
            ),
            "constrained_mean_recall": float(
                group_df["constrained_recall"].mean()
            ),
            "unconstrained_mean_area_ratio": float(
                group_df["unconstrained_area_ratio"].mean()
            ),
            "constrained_mean_area_ratio": float(
                group_df["constrained_area_ratio"].mean()
            ),
            "mean_dice_change": float(
                group_df["dice_change"].mean()
            ),
        })

    group_df = pd.DataFrame(group_rows)

    group_df.to_csv(
        OUTPUT_DIR / "reference_area_group_performance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Existing annotation-group performance.
    # --------------------------------------------------------

    annotation_rows = []

    for group_name, group in df.groupby(
        "annotation_group",
        observed=True,
    ):

        annotation_rows.append({
            "annotation_group": str(group_name),
            "image_count": int(len(group)),
            "mean_reference_area_percentage": float(
                group["reference_infection_area_percentage"].mean()
            ),
            "unconstrained_mean_dice": float(
                group["unconstrained_dice"].mean()
            ),
            "constrained_mean_dice": float(
                group["constrained_dice"].mean()
            ),
            "mean_dice_change": float(
                group["dice_change"].mean()
            ),
        })

    annotation_df = pd.DataFrame(annotation_rows)

    annotation_df.to_csv(
        OUTPUT_DIR / "existing_annotation_group_performance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Correlation analysis.
    # --------------------------------------------------------

    correlation_results = {
        "reference_area_vs_unconstrained_dice": safe_spearman(
            df["reference_infection_area_percentage"],
            df["unconstrained_dice"],
        ),
        "reference_area_vs_constrained_dice": safe_spearman(
            df["reference_infection_area_percentage"],
            df["constrained_dice"],
        ),
        "reference_area_vs_unconstrained_precision": safe_spearman(
            df["reference_infection_area_percentage"],
            df["unconstrained_precision"],
        ),
        "reference_area_vs_unconstrained_recall": safe_spearman(
            df["reference_infection_area_percentage"],
            df["unconstrained_recall"],
        ),
        "reference_area_vs_unconstrained_area_ratio": safe_spearman(
            df["reference_infection_area_percentage"],
            df["unconstrained_area_ratio"],
        ),
    }

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    for name, result in correlation_results.items():
        print(f"\n{name}")
        print(result)

    # --------------------------------------------------------
    # Small vs non-small annotation comparison.
    # --------------------------------------------------------

    small = df[
        df["reference_area_group"].isin(
            ["VERY_SMALL", "SMALL"]
        )
    ]

    non_small = df[
        ~df["reference_area_group"].isin(
            ["VERY_SMALL", "SMALL"]
        )
    ]

    small_comparison = {}

    if len(small) >= 2 and len(non_small) >= 2:

        test = mannwhitneyu(
            small["unconstrained_dice"],
            non_small["unconstrained_dice"],
            alternative="two-sided",
        )

        small_comparison = {
            "small_image_count": int(len(small)),
            "non_small_image_count": int(len(non_small)),
            "small_mean_dice": float(
                small["unconstrained_dice"].mean()
            ),
            "non_small_mean_dice": float(
                non_small["unconstrained_dice"].mean()
            ),
            "mann_whitney_u": float(test.statistic),
            "p_value": float(test.pvalue),
        }

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("REFERENCE INFECTION AREA SUMMARY")
    print("=" * 80)

    print(
        f"Images analyzed: {len(df)}"
    )
    print(
        f"Mean infection area: "
        f"{df['reference_infection_area_percentage'].mean():.4f}%"
    )
    print(
        f"Median infection area: "
        f"{df['reference_infection_area_percentage'].median():.4f}%"
    )
    print(
        f"Minimum infection area: "
        f"{df['reference_infection_area_percentage'].min():.4f}%"
    )
    print(
        f"Maximum infection area: "
        f"{df['reference_infection_area_percentage'].max():.4f}%"
    )

    print("\n" + "=" * 80)
    print("REFERENCE AREA GROUP PERFORMANCE")
    print("=" * 80)

    print(group_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("EXISTING ANNOTATION GROUP PERFORMANCE")
    print("=" * 80)

    print(annotation_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("SMALL VS NON-SMALL COMPARISON")
    print("=" * 80)

    print(
        json.dumps(
            small_comparison,
            indent=4,
        )
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    summary = {
        "study": "Infection area and segmentation performance",
        "image_count": int(len(df)),
        "dice_reconstruction": {
            "max_unconstrained_difference":
                max_unconstrained_difference,
            "max_constrained_difference":
                max_constrained_difference,
        },
        "reference_area_statistics": {
            "mean_percentage": float(
                df["reference_infection_area_percentage"].mean()
            ),
            "median_percentage": float(
                df["reference_infection_area_percentage"].median()
            ),
            "minimum_percentage": float(
                df["reference_infection_area_percentage"].min()
            ),
            "maximum_percentage": float(
                df["reference_infection_area_percentage"].max()
            ),
        },
        "reference_area_group_results": group_rows,
        "existing_annotation_group_results": annotation_rows,
        "correlations": correlation_results,
        "small_vs_non_small": small_comparison,
        "limitations": [
            "Analysis uses one validation dataset.",
            "Reference infection area describes annotation geometry.",
            "Annotation area is not a clinical severity measurement.",
            "Correlation does not establish causation.",
            "COVID-QU-Ex infection masks are used; this analysis "
            "does not establish performance on bacterial or viral "
            "infection segmentation.",
            "Some reference masks are identical to lung masks.",
        ],
    }

    with open(
        OUTPUT_DIR / "infection_area_performance_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 80)
    print("SCRIPT 106 COMPLETED")
    print("=" * 80)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. infection_area_per_image_analysis.csv")
    print("2. reference_area_group_performance.csv")
    print("3. existing_annotation_group_performance.csv")
    print("4. infection_area_performance_summary.json")


if __name__ == "__main__":
    main()
