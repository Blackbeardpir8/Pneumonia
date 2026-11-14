
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import (
    CLASSIFICATION_RESULTS_DIR,
    CLASS_NAMES,
)

# ============================================================
# 2. PATHS
# ============================================================

PREDICTIONS_PATH = (
    CLASSIFICATION_RESULTS_DIR
    / "validation_evaluation"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    CLASSIFICATION_RESULTS_DIR
    / "error_analysis"
)


# ============================================================
# 3. LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "processed_path",
        "class_name",
        "class_index",
        "actual_class",
        "predicted_class",
        "correct_prediction",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if "source_dataset" not in df.columns:
        raise ValueError(
            "source_dataset is missing. "
            "Regenerate validation_predictions.csv with source_dataset."
        )

    df["correct_prediction"] = (
        df["correct_prediction"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["correct_prediction"].isna().any():
        raise ValueError(
            "Invalid values found in correct_prediction."
        )

    # Independently verify prediction correctness.
    calculated_correct = (
        df["actual_class"] == df["predicted_class"]
    )

    if not np.array_equal(
        calculated_correct.to_numpy(),
        df["correct_prediction"].to_numpy(),
    ):
        raise ValueError(
            "Prediction correctness does not match actual labels."
        )

    print("\nValidation predictions loaded.")
    print("Total predictions:", len(df))

    return df


# ============================================================
# 4. GENERAL ERROR ANALYSIS
# ============================================================

def analyze_general_errors(df):

    errors = df[
        ~df["correct_prediction"]
    ].copy()

    print("\nGENERAL ERROR ANALYSIS")
    print("-" * 65)

    print("Total validation images:", len(df))
    print("Correct predictions:", len(df) - len(errors))
    print("Incorrect predictions:", len(errors))

    print("\nErrors by actual class:")

    print(
        errors["actual_class"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    print("\nErrors by predicted class:")

    print(
        errors["predicted_class"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    return errors


# ============================================================
# 5. ANALYZE CONFUSION PAIRS
# ============================================================

def analyze_confusion_pairs(errors):

    confusion_pairs = (
        errors.groupby(
            ["actual_class", "predicted_class"]
        )
        .size()
        .reset_index(name="error_count")
        .sort_values(
            "error_count",
            ascending=False,
        )
    )

    print("\nTOP CLASSIFICATION CONFUSIONS")
    print("-" * 65)

    print(confusion_pairs.to_string(index=False))

    return confusion_pairs


# ============================================================
# 6. ANALYZE ERRORS BY DATASET SOURCE
# ============================================================

def analyze_source_errors(errors):

    print("\nERRORS BY SOURCE DATASET")
    print("-" * 65)

    source_errors = (
        errors.groupby(
            [
                "actual_class",
                "predicted_class",
                "source_dataset",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="error_count")
        .sort_values(
            "error_count",
            ascending=False,
        )
    )

    print(source_errors.to_string(index=False))

    # Focus on Viral -> Bacterial errors.
    viral_to_bacterial = errors[
        (errors["actual_class"] == "Viral")
        & (errors["predicted_class"] == "Bacterial")
    ]

    print("\nVIRAL PREDICTED AS BACTERIAL")
    print("-" * 65)

    print(
        "Total:",
        len(viral_to_bacterial),
    )

    print("\nSource-wise breakdown:")

    viral_source_counts = (
        viral_to_bacterial["source_dataset"]
        .value_counts(dropna=False)
    )

    print(viral_source_counts.to_string())

    # Calculate source-specific error rates using all
    # actual Viral validation images from each source.
    viral_all = errors.copy()

    # This function receives only incorrect predictions,
    # so the complete validation dataframe is needed
    # for accurate source-specific error rates.
    return source_errors, viral_to_bacterial


# ============================================================
# 7. SOURCE-WISE VIRAL ERROR RATES
# ============================================================

def calculate_viral_source_error_rates(df):

    viral_df = df[
        df["actual_class"] == "Viral"
    ].copy()

    if viral_df.empty:
        return pd.DataFrame()

    source_summary = (
        viral_df.groupby("source_dataset", dropna=False)
        .agg(
            total_viral_images=("actual_class", "size"),
            correctly_classified=("correct_prediction", "sum"),
        )
        .reset_index()
    )

    source_summary["incorrect_predictions"] = (
        source_summary["total_viral_images"]
        - source_summary["correctly_classified"]
    )

    source_summary["viral_to_bacterial_errors"] = (
        viral_df[
            viral_df["predicted_class"] == "Bacterial"
        ]
        .groupby("source_dataset", dropna=False)
        .size()
        .reindex(
            source_summary["source_dataset"]
        )
        .fillna(0)
        .to_numpy()
    )

    source_summary["accuracy_percentage"] = (
        source_summary["correctly_classified"]
        / source_summary["total_viral_images"]
        * 100
    )

    source_summary["viral_to_bacterial_error_percentage"] = (
        source_summary["viral_to_bacterial_errors"]
        / source_summary["total_viral_images"]
        * 100
    )

    source_summary = source_summary.sort_values(
        "total_viral_images",
        ascending=False,
    )

    print("\nVIRAL CLASSIFICATION BY SOURCE")
    print("-" * 65)

    print(source_summary.to_string(index=False))

    return source_summary


# ============================================================
# 8. ANALYZE PREDICTION CONFIDENCE
# ============================================================

def analyze_prediction_confidence(errors):

    bacterial_probability_column = "probability_Bacterial"
    viral_probability_column = "probability_Viral"

    required = [
        bacterial_probability_column,
        viral_probability_column,
    ]

    if not all(column in errors.columns for column in required):
        print("\nProbability columns unavailable.")
        return pd.DataFrame()

    viral_to_bacterial = errors[
        (errors["actual_class"] == "Viral")
        & (errors["predicted_class"] == "Bacterial")
    ].copy()

    if viral_to_bacterial.empty:
        return pd.DataFrame()

    viral_to_bacterial["confidence_difference"] = (
        viral_to_bacterial[bacterial_probability_column]
        - viral_to_bacterial[viral_probability_column]
    )

    print("\nVIRAL TO BACTERIAL PREDICTION CONFIDENCE")
    print("-" * 65)

    print(
        viral_to_bacterial[
            [
                "actual_class",
                "predicted_class",
                bacterial_probability_column,
                viral_probability_column,
                "confidence_difference",
            ]
        ].describe().round(4)
    )

    return viral_to_bacterial


# ============================================================
# 9. SAVE REPORTS
# ============================================================

def save_reports(
    errors,
    confusion_pairs,
    source_errors,
    source_summary,
    viral_to_bacterial,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors.to_csv(
        OUTPUT_DIR / "all_misclassified_images.csv",
        index=False,
    )

    confusion_pairs.to_csv(
        OUTPUT_DIR / "confusion_pairs.csv",
        index=False,
    )

    source_errors.to_csv(
        OUTPUT_DIR / "errors_by_source.csv",
        index=False,
    )

    source_summary.to_csv(
        OUTPUT_DIR / "viral_source_error_rates.csv",
        index=False,
    )

    viral_to_bacterial.to_csv(
        OUTPUT_DIR / "viral_predicted_as_bacterial.csv",
        index=False,
    )

    print("\nReports saved in:")
    print(OUTPUT_DIR)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PNEUMONIA CDSS - CLASSIFICATION ERROR ANALYSIS")
    print("=" * 70)

    df = load_predictions()

    errors = analyze_general_errors(df)

    confusion_pairs = analyze_confusion_pairs(errors)

    source_errors, viral_to_bacterial = (
        analyze_source_errors(errors)
    )

    source_summary = calculate_viral_source_error_rates(df)

    confidence_errors = analyze_prediction_confidence(errors)

    save_reports(
        errors,
        confusion_pairs,
        source_errors,
        source_summary,
        confidence_errors,
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION ERROR ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
