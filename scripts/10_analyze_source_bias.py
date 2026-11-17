
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiml.config import CLASS_NAMES


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "validation_evaluation"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "source_bias_analysis"
)

CLASS_ORDER = list(CLASS_NAMES)

PROBABILITY_COLUMNS = [
    f"probability_{class_name}"
    for class_name in CLASS_ORDER
]


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_predictions():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "source_dataset",
        "actual_class",
        "predicted_class",
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

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    unknown_actual = set(df["actual_class"]) - set(CLASS_ORDER)
    unknown_predicted = set(df["predicted_class"]) - set(CLASS_ORDER)

    if unknown_actual or unknown_predicted:
        raise ValueError(
            f"Unknown actual classes: {unknown_actual}\n"
            f"Unknown predicted classes: {unknown_predicted}"
        )

    missing_probabilities = [
        column
        for column in PROBABILITY_COLUMNS
        if column not in df.columns
    ]

    if missing_probabilities:
        raise ValueError(
            "Missing probability columns: "
            f"{missing_probabilities}"
        )

    for column in PROBABILITY_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[PROBABILITY_COLUMNS].isnull().any().any():
        raise ValueError(
            "Probability columns contain invalid or missing values."
        )

    df["correct_prediction"] = (
        df["actual_class"] == df["predicted_class"]
    )

    return df


# --------------------------------------------------
# SOURCE-WISE SUMMARY
# --------------------------------------------------

def analyze_source_accuracy(df):

    results = []

    for source, group in df.groupby("source_dataset"):

        accuracy = accuracy_score(
            group["actual_class"],
            group["predicted_class"]
        )

        results.append({
            "source_dataset": source,
            "total_images": len(group),
            "correct_predictions": int(
                group["correct_prediction"].sum()
            ),
            "incorrect_predictions": int(
                (~group["correct_prediction"]).sum()
            ),
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# CLASS-WISE PERFORMANCE BY SOURCE
# --------------------------------------------------

def analyze_class_performance(df):

    results = []

    for source, group in df.groupby("source_dataset"):

        report = classification_report(
            group["actual_class"],
            group["predicted_class"],
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )

        for class_name in CLASS_ORDER:

            metrics = report[class_name]

            actual_count = int(
                (group["actual_class"] == class_name).sum()
            )

            results.append({
                "source_dataset": source,
                "class_name": class_name,
                "support": actual_count,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# SOURCE-WISE CONFUSION MATRICES
# --------------------------------------------------

def analyze_confusion_matrices(df):

    results = []

    for source, group in df.groupby("source_dataset"):

        matrix = confusion_matrix(
            group["actual_class"],
            group["predicted_class"],
            labels=CLASS_ORDER,
        )

        for actual_index, actual_class in enumerate(CLASS_ORDER):

            for predicted_index, predicted_class in enumerate(CLASS_ORDER):

                results.append({
                    "source_dataset": source,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "image_count": int(
                        matrix[actual_index, predicted_index]
                    ),
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# ACTUAL CLASS DISTRIBUTION BY SOURCE
# --------------------------------------------------

def analyze_class_distribution(df):

    distribution = (
        df.groupby(
            ["source_dataset", "actual_class"]
        )
        .size()
        .reset_index(name="image_count")
    )

    return distribution


# --------------------------------------------------
# AVERAGE PREDICTED PROBABILITIES
# --------------------------------------------------

def analyze_average_probabilities(df):

    results = []

    grouped = df.groupby(
        ["source_dataset", "actual_class"]
    )

    for (source, actual_class), group in grouped:

        row = {
            "source_dataset": source,
            "actual_class": actual_class,
            "image_count": len(group),
        }

        for column in PROBABILITY_COLUMNS:

            row[f"mean_{column}"] = group[column].mean()
            row[f"median_{column}"] = group[column].median()

        results.append(row)

    return pd.DataFrame(results)


# --------------------------------------------------
# TOP CONFUSION PAIRS BY SOURCE
# --------------------------------------------------

def analyze_top_confusions(df):

    errors = df[
        df["actual_class"] != df["predicted_class"]
    ]

    if errors.empty:
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "actual_class",
                "predicted_class",
                "error_count",
            ]
        )

    results = (
        errors.groupby([
            "source_dataset",
            "actual_class",
            "predicted_class",
        ])
        .size()
        .reset_index(name="error_count")
        .sort_values(
            ["source_dataset", "error_count"],
            ascending=[True, False]
        )
    )

    return results


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_csv(dataframe, filename):

    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading validation predictions...")

    df = load_predictions()

    print(f"Total validation images: {len(df)}")
    print(
        "Dataset sources:",
        df["source_dataset"].nunique()
    )

    print("\nAnalyzing source-wise accuracy...")

    source_accuracy = analyze_source_accuracy(df)

    print(source_accuracy.to_string(index=False))

    print("\nAnalyzing class-wise performance...")

    class_performance = analyze_class_performance(df)

    print(class_performance.to_string(index=False))

    print("\nAnalyzing confusion matrices...")

    confusion_results = analyze_confusion_matrices(df)

    print("\nAnalyzing class distribution...")

    class_distribution = analyze_class_distribution(df)

    print("\nAnalyzing predicted probabilities...")

    probability_results = analyze_average_probabilities(df)

    print("\nAnalyzing top confusion pairs...")

    top_confusions = analyze_top_confusions(df)

    # Save reports

    save_csv(
        source_accuracy,
        "source_accuracy.csv"
    )

    save_csv(
        class_performance,
        "class_performance_by_source.csv"
    )

    save_csv(
        confusion_results,
        "confusion_matrices_by_source.csv"
    )

    save_csv(
        class_distribution,
        "class_distribution_by_source.csv"
    )

    save_csv(
        probability_results,
        "average_probabilities_by_source.csv"
    )

    save_csv(
        top_confusions,
        "top_confusions_by_source.csv"
    )

    # Save general summary

    summary = {
        "total_validation_images": int(len(df)),
        "dataset_sources": sorted(
            df["source_dataset"].unique().tolist()
        ),
        "classes": CLASS_ORDER,
        "overall_accuracy": float(
            accuracy_score(
                df["actual_class"],
                df["predicted_class"]
            )
        ),
        "overall_accuracy_percent": float(
            accuracy_score(
                df["actual_class"],
                df["predicted_class"]
            ) * 100
        ),
        "source_accuracy": source_accuracy.to_dict(
            orient="records"
        ),
    }

    summary_path = OUTPUT_DIR / "analysis_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print(f"\nSaved: {summary_path}")

    print("\n" + "=" * 55)
    print("SOURCE-WISE ANALYSIS COMPLETED")
    print("=" * 55)

    print(f"Total images: {len(df)}")
    print(f"Overall accuracy: {summary['overall_accuracy_percent']:.2f}%")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/10_analyze_source_bias.py