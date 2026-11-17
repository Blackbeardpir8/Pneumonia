
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = (
    PROJECT_ROOT
    / "results/classification/lung_background_classifier_ablation"
    / "ablation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/classification/source_wise_confusion_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

CLASS_NAMES = [
    "Normal",
    "Viral",
    "Bacterial",
    "COVID",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

CONDITIONS = [
    "original",
    "lung_only",
    "background_only",
]


# =========================================================
# VALIDATE INPUT
# =========================================================

def validate_data(df):

    required_columns = [
        "actual_class",
        "source_dataset",
    ]

    for condition in CONDITIONS:
        required_columns.append(
            f"{condition}_predicted_class"
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    invalid_actual = (
        set(df["actual_class"]) - set(CLASS_NAMES)
    )

    if invalid_actual:
        raise ValueError(
            f"Invalid actual classes: {invalid_actual}"
        )

    for condition in CONDITIONS:

        column = f"{condition}_predicted_class"

        invalid_predictions = (
            set(df[column]) - set(CLASS_NAMES)
        )

        if invalid_predictions:
            raise ValueError(
                f"Invalid predictions in {column}: "
                f"{invalid_predictions}"
            )


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():

    print("=" * 65)
    print("SOURCE-WISE CONFUSION MATRIX ANALYSIS")
    print("=" * 65)

    print("\nLoading ablation predictions...")

    df = pd.read_csv(INPUT_CSV)

    validate_data(df)

    print(f"Total validation records: {len(df)}")

    sources = sorted(
        df["source_dataset"].unique()
    )

    print("\nDataset sources:")

    for source in sources:

        count = int(
            (df["source_dataset"] == source).sum()
        )

        print(f"{source}: {count}")

    all_reports = {}
    all_matrices = {}
    summary_rows = []

    # -----------------------------------------------------
    # Analyze each source separately
    # -----------------------------------------------------

    for source in sources:

        source_df = df[
            df["source_dataset"] == source
        ].copy()

        y_true = np.array([
            CLASS_TO_INDEX[name]
            for name in source_df["actual_class"]
        ])

        source_name = (
            source.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        print("\n" + "=" * 65)
        print(f"SOURCE: {source}")
        print(f"Images: {len(source_df)}")
        print("=" * 65)

        all_reports[source] = {}
        all_matrices[source] = {}

        # -------------------------------------------------
        # Analyze three conditions
        # -------------------------------------------------

        for condition in CONDITIONS:

            prediction_column = (
                f"{condition}_predicted_class"
            )

            y_pred = np.array([
                CLASS_TO_INDEX[name]
                for name in source_df[prediction_column]
            ])

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=list(range(len(CLASS_NAMES)))
            )

            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0
            )

            all_reports[source][condition] = report

            all_matrices[source][condition] = (
                matrix.tolist()
            )

            summary_rows.append({
                "source_dataset": source,
                "condition": condition,
                "total_images": len(source_df),
                "accuracy": accuracy,
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            })

            print(f"\n--- {condition.upper()} ---")

            print(f"Accuracy: {accuracy * 100:.2f}%")

            print("\nConfusion matrix:")
            print(matrix)

            print("\nClassification report:")

            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(len(CLASS_NAMES))),
                    target_names=CLASS_NAMES,
                    zero_division=0
                )
            )

    # -----------------------------------------------------
    # Save confusion matrices
    # -----------------------------------------------------

    with open(
        OUTPUT_DIR / "source_wise_confusion_matrices.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_matrices,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Save classification reports
    # -----------------------------------------------------

    with open(
        OUTPUT_DIR / "source_wise_classification_reports.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_reports,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "source_wise_metrics_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # Save readable confusion matrix CSV files
    # -----------------------------------------------------

    for source in sources:

        source_name = (
            source.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        for condition in CONDITIONS:

            matrix = np.array(
                all_matrices[source][condition]
            )

            matrix_df = pd.DataFrame(
                matrix,
                index=[
                    f"Actual_{name}"
                    for name in CLASS_NAMES
                ],
                columns=[
                    f"Predicted_{name}"
                    for name in CLASS_NAMES
                ]
            )

            filename = (
                f"{source_name}_{condition}_confusion_matrix.csv"
            )

            matrix_df.to_csv(
                OUTPUT_DIR / filename
            )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("SOURCE-WISE METRICS SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nAnalysis completed.")

    print(f"Output folder: {OUTPUT_DIR}")

    print("\nGenerated files:")
    print("- source_wise_confusion_matrices.json")
    print("- source_wise_classification_reports.json")
    print("- source_wise_metrics_summary.csv")
    print("- Six individual confusion matrix CSV files")


if __name__ == "__main__":
    main()

# to run this script, use the following command:
# python scripts/20_source_wise_confusion_analysis.py